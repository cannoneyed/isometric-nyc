# Task 031 deployment plan
_Exported on 12/18/2025 at 15:39:37 EST from Cursor (2.2.20)_

---

**User**

Implement the plan in @tasks/031_deploy.md 

---

**Cursor**

I'll implement the deployment script. Let me first explore the `src/app` directory to understand the project structure and build system.

Now I have a clear picture. Let me create the deployment script. I'll need to add `boto3` for R2 uploads since Cloudflare R2 is S3-compatible.

```python
#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing pmtiles to Cloudflare R2
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Try to import boto3 for R2 uploads
try:
  import boto3
  from botocore.config import Config

  HAS_BOTO3 = True
except ImportError:
  HAS_BOTO3 = False

# Configuration
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.parent
DIST_DIR = APP_DIR / "dist"
PUBLIC_DIR = APP_DIR / "public"

# GitHub Pages configuration
GITHUB_PAGES_BRANCH = "gh-pages"
GITHUB_PAGES_REMOTE = "origin"

# Cloudflare R2 configuration
R2_BUCKET_NAME = "isometric-nyc"
R2_ENDPOINT_URL = os.environ.get(
  "R2_ENDPOINT_URL", "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
)
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")

# Public URL where R2 content will be accessible
R2_PUBLIC_URL = os.environ.get(
  "R2_PUBLIC_URL", "https://isometric-nyc.r2.dev"
)


def run_command(
  cmd: list[str],
  cwd: Path | None = None,
  check: bool = True,
  capture_output: bool = False,
) -> subprocess.CompletedProcess:
  """Run a shell command and handle errors."""
  print(f"  → Running: {' '.join(cmd)}")
  result = subprocess.run(
    cmd,
    cwd=cwd,
    check=check,
    capture_output=capture_output,
    text=True,
  )
  return result


def build_app() -> bool:
  """Build the static web app using bun/vite."""
  print("\n📦 Building web app...")

  # Check if bun is available
  if shutil.which("bun") is None:
    print("  ❌ Error: bun is not installed or not in PATH")
    print("  Install bun: curl -fsSL https://bun.sh/install | bash")
    return False

  # Install dependencies if needed
  if not (APP_DIR / "node_modules").exists():
    print("  📥 Installing dependencies...")
    run_command(["bun", "install"], cwd=APP_DIR)

  # Run the build
  print("  🔨 Running build...")
  run_command(["bun", "run", "build"], cwd=APP_DIR)

  if DIST_DIR.exists():
    print(f"  ✅ Build complete: {DIST_DIR}")
    return True
  else:
    print("  ❌ Build failed: dist directory not created")
    return False


def deploy_github_pages(dry_run: bool = False) -> bool:
  """Deploy built content to GitHub Pages branch."""
  print("\n🌐 Deploying to GitHub Pages...")

  if not DIST_DIR.exists():
    print("  ❌ Error: dist directory does not exist. Run build first.")
    return False

  # Create a temporary directory for the gh-pages content
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Copy dist contents to temp directory (excluding large assets)
    print("  📋 Preparing deployment files...")
    for item in DIST_DIR.iterdir():
      # Skip tiles and pmtiles - they'll be served from R2
      if item.name in ("tiles", "tiles_processed", "water_masks") or item.suffix == ".pmtiles":
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      dest = tmp_path / item.name
      if item.is_dir():
        shutil.copytree(item, dest)
      else:
        shutil.copy2(item, dest)

    # Create .nojekyll file to disable Jekyll processing
    (tmp_path / ".nojekyll").touch()

    # Create CNAME file if a custom domain is configured
    custom_domain = os.environ.get("GITHUB_PAGES_DOMAIN")
    if custom_domain:
      (tmp_path / "CNAME").write_text(custom_domain)
      print(f"    📝 Created CNAME for {custom_domain}")

    if dry_run:
      print("  🔍 Dry run - would deploy these files:")
      for f in tmp_path.rglob("*"):
        if f.is_file():
          print(f"    - {f.relative_to(tmp_path)}")
      return True

    # Initialize git repo in temp directory
    print("  🔧 Initializing git repository...")
    run_command(["git", "init"], cwd=tmp_path)
    run_command(["git", "checkout", "-b", GITHUB_PAGES_BRANCH], cwd=tmp_path)

    # Get the remote URL from the main repo
    result = run_command(
      ["git", "remote", "get-url", GITHUB_PAGES_REMOTE],
      cwd=PROJECT_ROOT,
      capture_output=True,
    )
    remote_url = result.stdout.strip()

    run_command(["git", "remote", "add", GITHUB_PAGES_REMOTE, remote_url], cwd=tmp_path)

    # Add and commit all files
    run_command(["git", "add", "-A"], cwd=tmp_path)
    run_command(
      ["git", "commit", "-m", "Deploy to GitHub Pages"],
      cwd=tmp_path,
    )

    # Force push to gh-pages branch
    print("  🚀 Pushing to GitHub Pages...")
    run_command(
      ["git", "push", "--force", GITHUB_PAGES_REMOTE, GITHUB_PAGES_BRANCH],
      cwd=tmp_path,
    )

    print("  ✅ GitHub Pages deployment complete!")
    return True


def upload_to_r2(dry_run: bool = False) -> bool:
  """Upload pmtiles and tile assets to Cloudflare R2."""
  print("\n☁️  Uploading to Cloudflare R2...")

  if not HAS_BOTO3:
    print("  ❌ Error: boto3 is not installed")
    print("  Install with: uv add boto3")
    return False

  # Check for required credentials
  if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
    print("  ❌ Error: R2 credentials not configured")
    print("  Set these environment variables:")
    print("    - R2_ENDPOINT_URL (e.g., https://<account_id>.r2.cloudflarestorage.com)")
    print("    - R2_ACCESS_KEY_ID")
    print("    - R2_SECRET_ACCESS_KEY")
    return False

  if "<ACCOUNT_ID>" in R2_ENDPOINT_URL:
    print("  ❌ Error: R2_ENDPOINT_URL not configured")
    print("  Set R2_ENDPOINT_URL to your Cloudflare R2 endpoint")
    return False

  # Find pmtiles file
  pmtiles_file = PUBLIC_DIR / "tiles.pmtiles"
  if not pmtiles_file.exists():
    pmtiles_file = DIST_DIR / "tiles.pmtiles"

  if not pmtiles_file.exists():
    print(f"  ⚠️  Warning: tiles.pmtiles not found in {PUBLIC_DIR} or {DIST_DIR}")
    pmtiles_file = None

  # Collect files to upload
  files_to_upload: list[tuple[Path, str]] = []

  if pmtiles_file:
    files_to_upload.append((pmtiles_file, "tiles.pmtiles"))

  # Add tile directories if they exist
  for tile_dir_name in ["tiles", "tiles_processed", "water_masks"]:
    tile_dir = PUBLIC_DIR / tile_dir_name
    if tile_dir.exists():
      for file_path in tile_dir.rglob("*"):
        if file_path.is_file():
          relative_path = file_path.relative_to(PUBLIC_DIR)
          files_to_upload.append((file_path, str(relative_path)))

  if not files_to_upload:
    print("  ⚠️  No files found to upload")
    return True

  print(f"  📊 Found {len(files_to_upload)} files to upload")

  if dry_run:
    print("  🔍 Dry run - would upload these files:")
    for local_path, remote_key in files_to_upload[:20]:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    - {remote_key} ({size_mb:.2f} MB)")
    if len(files_to_upload) > 20:
      print(f"    ... and {len(files_to_upload) - 20} more files")
    return True

  # Create S3 client for R2
  s3_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(
      signature_version="s3v4",
      retries={"max_attempts": 3, "mode": "adaptive"},
    ),
  )

  # Upload files
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      # Determine content type
      content_type = get_content_type(local_path)

      print(f"    ⬆️  Uploading {remote_key}...", end=" ", flush=True)
      s3_client.upload_file(
        str(local_path),
        R2_BUCKET_NAME,
        remote_key,
        ExtraArgs={"ContentType": content_type},
      )
      print("✓")
      uploaded += 1
    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  print(f"\n  📊 Upload summary: {uploaded} succeeded, {failed} failed")

  if failed == 0:
    print("  ✅ R2 upload complete!")
    if pmtiles_file:
      print(f"  🔗 PMTiles URL: {R2_PUBLIC_URL}/tiles.pmtiles")
    return True
  else:
    print("  ⚠️  Some uploads failed")
    return False


def get_content_type(file_path: Path) -> str:
  """Get the content type for a file based on its extension."""
  content_types = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".pmtiles": "application/octet-stream",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
  }
  return content_types.get(file_path.suffix.lower(), "application/octet-stream")


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Environment Variables:
  R2_ENDPOINT_URL       Cloudflare R2 endpoint URL
  R2_ACCESS_KEY_ID      Cloudflare R2 access key ID
  R2_SECRET_ACCESS_KEY  Cloudflare R2 secret access key
  R2_PUBLIC_URL         Public URL for R2 content (default: https://isometric-nyc.r2.dev)
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages (optional)

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run
    """,
  )

  parser.add_argument(
    "--build",
    action="store_true",
    help="Build the web app",
  )
  parser.add_argument(
    "--github-pages",
    action="store_true",
    help="Deploy to GitHub Pages",
  )
  parser.add_argument(
    "--r2",
    action="store_true",
    help="Upload to Cloudflare R2",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )

  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all:
    if not build_app():
      success = False
      if not args.all:
        return 1

  # GitHub Pages
  if args.github_pages or args.all:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  # R2 upload
  if args.r2 or args.all:
    if not upload_to_r2(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  print("\n" + "=" * 40)
  if success:
    print("✅ Deployment complete!")
  else:
    print("⚠️  Deployment completed with some errors")

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

Now let me add boto3 as a dependency and check for linting errors:

The deployment script is now created at `src/app/deploy.py`. Here's what it does:

### Features

1. **Build** (`--build`): Runs `bun run build` to create the production static files in `dist/`

2. **GitHub Pages** (`--github-pages`): 
   - Copies built HTML/JS/CSS to a temporary git repo
   - Excludes tiles/pmtiles (they'll be served from R2)
   - Force pushes to the `gh-pages` branch

3. **Cloudflare R2** (`--r2`):
   - Uploads `tiles.pmtiles` and tile directories to the `isometric-nyc` bucket
   - Uses boto3 (S3-compatible API)

### Usage

```bash
# Build only
uv run python src/app/deploy.py --build

# Full deployment
uv run python src/app/deploy.py --all

# Dry run to see what would happen
uv run python src/app/deploy.py --all --dry-run
```

### Required Setup

1. **Add boto3 dependency** for R2 uploads:
   ```bash
   uv add boto3
   ```

2. **Set environment variables** for R2:
   ```bash
   export R2_ENDPOINT_URL="https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
   export R2_ACCESS_KEY_ID="your-access-key"
   export R2_SECRET_ACCESS_KEY="your-secret-key"
   export R2_PUBLIC_URL="https://isometric-nyc.r2.dev"  # optional
   ```

3. **Optional**: Set a custom domain for GitHub Pages:
   ```bash
   export GITHUB_PAGES_DOMAIN="your-custom-domain.com"
   ```

The script gracefully handles missing dependencies/credentials and will tell you what's needed if something isn't configured.

```shellscript
# Build only
uv run python src/app/deploy.py --build

# Full deployment
uv run python src/app/deploy.py --all

# Dry run to see what would happen
uv run python src/app/deploy.py --all --dry-run
```

```shellscript
   uv add boto3
```

```shellscript
   export R2_ENDPOINT_URL="https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
   export R2_ACCESS_KEY_ID="your-access-key"
   export R2_SECRET_ACCESS_KEY="your-secret-key"
   export R2_PUBLIC_URL="https://isometric-nyc.r2.dev"  # optional
```

```shellscript
   export GITHUB_PAGES_DOMAIN="your-custom-domain.com"
```

---

**User**

Why boto3? I'm using cloudflare R2

---

**Cursor**

Good point! Cloudflare R2 is S3-compatible, so boto3 works, but using Cloudflare's native tooling would be cleaner. There are two better options:

1. **Wrangler CLI** - Cloudflare's official CLI (probably already installed if you use Workers)
2. **cloudflare-python** - Cloudflare's official Python SDK

Would you prefer I rewrite the R2 upload to use **wrangler** (shell commands) or the **cloudflare Python SDK**? 

Wrangler is simpler if you already have it set up:
```bash
wrangler r2 object put isometric-nyc/tiles.pmtiles --file ./public/tiles.pmtiles
```

```shellscript
wrangler r2 object put isometric-nyc/tiles.pmtiles --file ./public/tiles.pmtiles
```

---

**User**

how do i install wrangler?

---

**Cursor**

Wrangler is installed via npm/bun:

```bash
# Using bun (since your project already uses it)
bun add -g wrangler

# Or using npm
npm install -g wrangler
```

Then authenticate:
```bash
wrangler login
```

This opens a browser to log into your Cloudflare account and authorize wrangler.

Want me to update the deploy script to use wrangler instead of boto3?

```shellscript
# Using bun (since your project already uses it)
bun add -g wrangler

# Or using npm
npm install -g wrangler
```

```shellscript
wrangler login
```

---

**User**

yes please

---

**Cursor**

```python
#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing pmtiles to Cloudflare R2 (using wrangler)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.parent
DIST_DIR = APP_DIR / "dist"
PUBLIC_DIR = APP_DIR / "public"

# GitHub Pages configuration
GITHUB_PAGES_BRANCH = "gh-pages"
GITHUB_PAGES_REMOTE = "origin"

# Cloudflare R2 configuration
R2_BUCKET_NAME = "isometric-nyc"


def run_command(
  cmd: list[str],
  cwd: Path | None = None,
  check: bool = True,
  capture_output: bool = False,
) -> subprocess.CompletedProcess:
  """Run a shell command and handle errors."""
  print(f"  → Running: {' '.join(cmd)}")
  result = subprocess.run(
    cmd,
    cwd=cwd,
    check=check,
    capture_output=capture_output,
    text=True,
  )
  return result


def build_app() -> bool:
  """Build the static web app using bun/vite."""
  print("\n📦 Building web app...")

  # Check if bun is available
  if shutil.which("bun") is None:
    print("  ❌ Error: bun is not installed or not in PATH")
    print("  Install bun: curl -fsSL https://bun.sh/install | bash")
    return False

  # Install dependencies if needed
  if not (APP_DIR / "node_modules").exists():
    print("  📥 Installing dependencies...")
    run_command(["bun", "install"], cwd=APP_DIR)

  # Run the build
  print("  🔨 Running build...")
  run_command(["bun", "run", "build"], cwd=APP_DIR)

  if DIST_DIR.exists():
    print(f"  ✅ Build complete: {DIST_DIR}")
    return True
  else:
    print("  ❌ Build failed: dist directory not created")
    return False


def deploy_github_pages(dry_run: bool = False) -> bool:
  """Deploy built content to GitHub Pages branch."""
  print("\n🌐 Deploying to GitHub Pages...")

  if not DIST_DIR.exists():
    print("  ❌ Error: dist directory does not exist. Run build first.")
    return False

  # Create a temporary directory for the gh-pages content
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Copy dist contents to temp directory (excluding large assets)
    print("  📋 Preparing deployment files...")
    for item in DIST_DIR.iterdir():
      # Skip tiles and pmtiles - they'll be served from R2
      if (
        item.name in ("tiles", "tiles_processed", "water_masks")
        or item.suffix == ".pmtiles"
      ):
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      dest = tmp_path / item.name
      if item.is_dir():
        shutil.copytree(item, dest)
      else:
        shutil.copy2(item, dest)

    # Create .nojekyll file to disable Jekyll processing
    (tmp_path / ".nojekyll").touch()

    # Create CNAME file if a custom domain is configured
    custom_domain = os.environ.get("GITHUB_PAGES_DOMAIN")
    if custom_domain:
      (tmp_path / "CNAME").write_text(custom_domain)
      print(f"    📝 Created CNAME for {custom_domain}")

    if dry_run:
      print("  🔍 Dry run - would deploy these files:")
      for f in tmp_path.rglob("*"):
        if f.is_file():
          print(f"    - {f.relative_to(tmp_path)}")
      return True

    # Initialize git repo in temp directory
    print("  🔧 Initializing git repository...")
    run_command(["git", "init"], cwd=tmp_path)
    run_command(["git", "checkout", "-b", GITHUB_PAGES_BRANCH], cwd=tmp_path)

    # Get the remote URL from the main repo
    result = run_command(
      ["git", "remote", "get-url", GITHUB_PAGES_REMOTE],
      cwd=PROJECT_ROOT,
      capture_output=True,
    )
    remote_url = result.stdout.strip()

    run_command(["git", "remote", "add", GITHUB_PAGES_REMOTE, remote_url], cwd=tmp_path)

    # Add and commit all files
    run_command(["git", "add", "-A"], cwd=tmp_path)
    run_command(
      ["git", "commit", "-m", "Deploy to GitHub Pages"],
      cwd=tmp_path,
    )

    # Force push to gh-pages branch
    print("  🚀 Pushing to GitHub Pages...")
    run_command(
      ["git", "push", "--force", GITHUB_PAGES_REMOTE, GITHUB_PAGES_BRANCH],
      cwd=tmp_path,
    )

    print("  ✅ GitHub Pages deployment complete!")
    return True


def upload_to_r2(dry_run: bool = False) -> bool:
  """Upload pmtiles and tile assets to Cloudflare R2 using wrangler."""
  print("\n☁️  Uploading to Cloudflare R2...")

  # Check if wrangler is available
  if shutil.which("wrangler") is None:
    print("  ❌ Error: wrangler is not installed or not in PATH")
    print("  Install wrangler: bun add -g wrangler")
    print("  Then authenticate: wrangler login")
    return False

  # Find pmtiles file
  pmtiles_file = PUBLIC_DIR / "tiles.pmtiles"
  if not pmtiles_file.exists():
    pmtiles_file = DIST_DIR / "tiles.pmtiles"

  if not pmtiles_file.exists():
    print(f"  ⚠️  Warning: tiles.pmtiles not found in {PUBLIC_DIR} or {DIST_DIR}")
    pmtiles_file = None

  # Collect files to upload
  files_to_upload: list[tuple[Path, str]] = []

  if pmtiles_file:
    files_to_upload.append((pmtiles_file, "tiles.pmtiles"))

  # Add tile directories if they exist
  for tile_dir_name in ["tiles", "tiles_processed", "water_masks"]:
    tile_dir = PUBLIC_DIR / tile_dir_name
    if tile_dir.exists():
      for file_path in tile_dir.rglob("*"):
        if file_path.is_file():
          relative_path = file_path.relative_to(PUBLIC_DIR)
          files_to_upload.append((file_path, str(relative_path)))

  if not files_to_upload:
    print("  ⚠️  No files found to upload")
    return True

  print(f"  📊 Found {len(files_to_upload)} files to upload")

  if dry_run:
    print("  🔍 Dry run - would upload these files:")
    for local_path, remote_key in files_to_upload[:20]:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    - {remote_key} ({size_mb:.2f} MB)")
    if len(files_to_upload) > 20:
      print(f"    ... and {len(files_to_upload) - 20} more files")
    return True

  # Upload files using wrangler
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      print(f"    ⬆️  Uploading {remote_key}...", end=" ", flush=True)

      # Use wrangler r2 object put
      result = subprocess.run(
        [
          "wrangler",
          "r2",
          "object",
          "put",
          f"{R2_BUCKET_NAME}/{remote_key}",
          "--file",
          str(local_path),
        ],
        capture_output=True,
        text=True,
      )

      if result.returncode == 0:
        print("✓")
        uploaded += 1
      else:
        print(f"✗ ({result.stderr.strip()})")
        failed += 1

    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  print(f"\n  📊 Upload summary: {uploaded} succeeded, {failed} failed")

  if failed == 0:
    print("  ✅ R2 upload complete!")
    if pmtiles_file:
      print(f"  🔗 PMTiles will be available at your R2 public URL/tiles.pmtiles")
    return True
  else:
    print("  ⚠️  Some uploads failed")
    return False


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Prerequisites:
  - bun: Install with `curl -fsSL https://bun.sh/install | bash`
  - wrangler: Install with `bun add -g wrangler`, then run `wrangler login`

Environment Variables (optional):
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run

  # Just upload to R2
  python deploy.py --r2
    """,
  )

  parser.add_argument(
    "--build",
    action="store_true",
    help="Build the web app",
  )
  parser.add_argument(
    "--github-pages",
    action="store_true",
    help="Deploy to GitHub Pages",
  )
  parser.add_argument(
    "--r2",
    action="store_true",
    help="Upload to Cloudflare R2",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )

  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all:
    if not build_app():
      success = False
      if not args.all:
        return 1

  # GitHub Pages
  if args.github_pages or args.all:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  # R2 upload
  if args.r2 or args.all:
    if not upload_to_r2(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  print("\n" + "=" * 40)
  if success:
    print("✅ Deployment complete!")
  else:
    print("⚠️  Deployment completed with some errors")

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

```python
#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing pmtiles to Cloudflare R2 (using wrangler)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.parent
DIST_DIR = APP_DIR / "dist"
PUBLIC_DIR = APP_DIR / "public"

# GitHub Pages configuration
GITHUB_PAGES_BRANCH = "gh-pages"
GITHUB_PAGES_REMOTE = "origin"

# Cloudflare R2 configuration
R2_BUCKET_NAME = "isometric-nyc"


def run_command(
  cmd: list[str],
  cwd: Path | None = None,
  check: bool = True,
  capture_output: bool = False,
) -> subprocess.CompletedProcess:
  """Run a shell command and handle errors."""
  print(f"  → Running: {' '.join(cmd)}")
  result = subprocess.run(
    cmd,
    cwd=cwd,
    check=check,
    capture_output=capture_output,
    text=True,
  )
  return result


def build_app() -> bool:
  """Build the static web app using bun/vite."""
  print("\n📦 Building web app...")

  # Check if bun is available
  if shutil.which("bun") is None:
    print("  ❌ Error: bun is not installed or not in PATH")
    print("  Install bun: curl -fsSL https://bun.sh/install | bash")
    return False

  # Install dependencies if needed
  if not (APP_DIR / "node_modules").exists():
    print("  📥 Installing dependencies...")
    run_command(["bun", "install"], cwd=APP_DIR)

  # Run the build
  print("  🔨 Running build...")
  run_command(["bun", "run", "build"], cwd=APP_DIR)

  if DIST_DIR.exists():
    print(f"  ✅ Build complete: {DIST_DIR}")
    return True
  else:
    print("  ❌ Build failed: dist directory not created")
    return False


def deploy_github_pages(dry_run: bool = False) -> bool:
  """Deploy built content to GitHub Pages branch."""
  print("\n🌐 Deploying to GitHub Pages...")

  if not DIST_DIR.exists():
    print("  ❌ Error: dist directory does not exist. Run build first.")
    return False

  # Create a temporary directory for the gh-pages content
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Copy dist contents to temp directory (excluding large assets)
    print("  📋 Preparing deployment files...")
    for item in DIST_DIR.iterdir():
      # Skip tiles and pmtiles - they'll be served from R2
      if (
        item.name in ("tiles", "tiles_processed", "water_masks")
        or item.suffix == ".pmtiles"
      ):
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      dest = tmp_path / item.name
      if item.is_dir():
        shutil.copytree(item, dest)
      else:
        shutil.copy2(item, dest)

    # Create .nojekyll file to disable Jekyll processing
    (tmp_path / ".nojekyll").touch()

    # Create CNAME file if a custom domain is configured
    custom_domain = os.environ.get("GITHUB_PAGES_DOMAIN")
    if custom_domain:
      (tmp_path / "CNAME").write_text(custom_domain)
      print(f"    📝 Created CNAME for {custom_domain}")

    if dry_run:
      print("  🔍 Dry run - would deploy these files:")
      for f in tmp_path.rglob("*"):
        if f.is_file():
          print(f"    - {f.relative_to(tmp_path)}")
      return True

    # Initialize git repo in temp directory
    print("  🔧 Initializing git repository...")
    run_command(["git", "init"], cwd=tmp_path)
    run_command(["git", "checkout", "-b", GITHUB_PAGES_BRANCH], cwd=tmp_path)

    # Get the remote URL from the main repo
    result = run_command(
      ["git", "remote", "get-url", GITHUB_PAGES_REMOTE],
      cwd=PROJECT_ROOT,
      capture_output=True,
    )
    remote_url = result.stdout.strip()

    run_command(["git", "remote", "add", GITHUB_PAGES_REMOTE, remote_url], cwd=tmp_path)

    # Add and commit all files
    run_command(["git", "add", "-A"], cwd=tmp_path)
    run_command(
      ["git", "commit", "-m", "Deploy to GitHub Pages"],
      cwd=tmp_path,
    )

    # Force push to gh-pages branch
    print("  🚀 Pushing to GitHub Pages...")
    run_command(
      ["git", "push", "--force", GITHUB_PAGES_REMOTE, GITHUB_PAGES_BRANCH],
      cwd=tmp_path,
    )

    print("  ✅ GitHub Pages deployment complete!")
    return True


def upload_to_r2(dry_run: bool = False) -> bool:
  """Upload pmtiles and tile assets to Cloudflare R2 using wrangler."""
  print("\n☁️  Uploading to Cloudflare R2...")

  # Check if wrangler is available
  if shutil.which("wrangler") is None:
    print("  ❌ Error: wrangler is not installed or not in PATH")
    print("  Install wrangler: bun add -g wrangler")
    print("  Then authenticate: wrangler login")
    return False

  # Find pmtiles file
  pmtiles_file = PUBLIC_DIR / "tiles.pmtiles"
  if not pmtiles_file.exists():
    pmtiles_file = DIST_DIR / "tiles.pmtiles"

  if not pmtiles_file.exists():
    print(f"  ⚠️  Warning: tiles.pmtiles not found in {PUBLIC_DIR} or {DIST_DIR}")
    pmtiles_file = None

  # Collect files to upload
  files_to_upload: list[tuple[Path, str]] = []

  if pmtiles_file:
    files_to_upload.append((pmtiles_file, "tiles.pmtiles"))

  # Add tile directories if they exist
  for tile_dir_name in ["tiles", "tiles_processed", "water_masks"]:
    tile_dir = PUBLIC_DIR / tile_dir_name
    if tile_dir.exists():
      for file_path in tile_dir.rglob("*"):
        if file_path.is_file():
          relative_path = file_path.relative_to(PUBLIC_DIR)
          files_to_upload.append((file_path, str(relative_path)))

  if not files_to_upload:
    print("  ⚠️  No files found to upload")
    return True

  print(f"  📊 Found {len(files_to_upload)} files to upload")

  if dry_run:
    print("  🔍 Dry run - would upload these files:")
    for local_path, remote_key in files_to_upload[:20]:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    - {remote_key} ({size_mb:.2f} MB)")
    if len(files_to_upload) > 20:
      print(f"    ... and {len(files_to_upload) - 20} more files")
    return True

  # Upload files using wrangler
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      print(f"    ⬆️  Uploading {remote_key}...", end=" ", flush=True)

      # Use wrangler r2 object put
      result = subprocess.run(
        [
          "wrangler",
          "r2",
          "object",
          "put",
          f"{R2_BUCKET_NAME}/{remote_key}",
          "--file",
          str(local_path),
        ],
        capture_output=True,
        text=True,
      )

      if result.returncode == 0:
        print("✓")
        uploaded += 1
      else:
        print(f"✗ ({result.stderr.strip()})")
        failed += 1

    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  print(f"\n  📊 Upload summary: {uploaded} succeeded, {failed} failed")

  if failed == 0:
    print("  ✅ R2 upload complete!")
    if pmtiles_file:
      print("  🔗 PMTiles will be available at your R2 public URL/tiles.pmtiles")
    return True
  else:
    print("  ⚠️  Some uploads failed")
    return False


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Prerequisites:
  - bun: Install with `curl -fsSL https://bun.sh/install | bash`
  - wrangler: Install with `bun add -g wrangler`, then run `wrangler login`

Environment Variables (optional):
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run

  # Just upload to R2
  python deploy.py --r2
    """,
  )

  parser.add_argument(
    "--build",
    action="store_true",
    help="Build the web app",
  )
  parser.add_argument(
    "--github-pages",
    action="store_true",
    help="Deploy to GitHub Pages",
  )
  parser.add_argument(
    "--r2",
    action="store_true",
    help="Upload to Cloudflare R2",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )

  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all:
    if not build_app():
      success = False
      if not args.all:
        return 1

  # GitHub Pages
  if args.github_pages or args.all:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  # R2 upload
  if args.r2 or args.all:
    if not upload_to_r2(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  print("\n" + "=" * 40)
  if success:
    print("✅ Deployment complete!")
  else:
    print("⚠️  Deployment completed with some errors")

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

Done! I've updated the script to use **wrangler** instead of boto3. The changes:

1. **Removed boto3 dependency** - no Python packages needed
2. **Uses `wrangler r2 object put`** to upload files to R2

### Prerequisites

```bash
# Install wrangler globally
bun add -g wrangler

# Authenticate with Cloudflare
wrangler login
```

### Usage

```bash
# Build only
uv run python src/app/deploy.py --build

# Upload to R2 only
uv run python src/app/deploy.py --r2

# Full deployment
uv run python src/app/deploy.py --all

# Dry run
uv run python src/app/deploy.py --all --dry-run
```

```shellscript
# Install wrangler globally
bun add -g wrangler

# Authenticate with Cloudflare
wrangler login
```

```shellscript
# Build only
uv run python src/app/deploy.py --build

# Upload to R2 only
uv run python src/app/deploy.py --r2

# Full deployment
uv run python src/app/deploy.py --all

# Dry run
uv run python src/app/deploy.py --all --dry-run
```

---

**User**

Finally, update the README.md doc to have a section on deployment

---

**Cursor**

```markdown
# Isometric NYC

This project aims to generate a massive isometric pixel art view of New York
City using the latest and greatest AI tools available.

## Project Structure

```
isometric-nyc/
├── src/
│   ├── isometric_nyc/       # Python tile generation pipeline
│   ├── app/                 # Web viewer (React + Deck.gl)
│   ├── water_shader_demo/   # Water shader demo (React + WebGL)
│   └── web/                 # 3D tiles viewer (Three.js)
├── synthetic_data/          # Training datasets
├── tile_plans/              # Tile generation plans
└── tasks/                   # Task documentation
```

---

## Web Viewer

A high-performance tiled map viewer for exploring the isometric pixel art map.

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **Deck.gl** — WebGL tile rendering with OrthographicView
- **Bun** — Package management

### Running the Web App

```bash
cd src/app

# Install dependencies
bun install

# Start development server
bun run dev
```

The app will open at **http://localhost:3000**

---

## Water Shader Demo

A WebGL shader demo for adding animated water effects to the isometric map
tiles.

### Features

- **Shoreline foam**: Subtle animated foam effect at water edges using a
  distance mask
- **Water ripples**: Gentle darkening ripples across open water areas
- **Adjustable parameters**: Real-time sliders for tweaking all effects
- **Pixel art aesthetic**: Quantized effects that maintain the retro look

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **WebGL** — Custom GLSL shaders for water animation
- **Bun** — Package management

### Running the Water Shader Demo

```bash
cd src/water_shader_demo

# Install dependencies
bun install

# Start development server
bun dev
```

The app will open at **http://localhost:5173** (or next available port)

### How It Works

The shader uses a **distance mask** approach:

- **Black pixels** in the mask = land (pass through unchanged)
- **White pixels** = deep water (receives ripple effects)
- **Gradient transition** = shoreline (receives foam effects)

The mask is a blurred version of the land/water boundary, providing smooth
distance information for wave animation.

### Shader Controls

| Control         | Description                        |
| --------------- | ---------------------------------- |
| Wave Speed      | Animation speed of shoreline waves |
| Wave Frequency  | Density of wave bands              |
| Foam Threshold  | How far from shore foam appears    |
| Water Darkness  | Overall brightness of water pixels |
| Ripple Darkness | Intensity of deep water ripples    |

### Adding Tile Assets

Place tile images and corresponding masks in:

```
src/water_shader_demo/public/
├── tiles/
│   └── 0_0.png          # The tile image
└── masks/
    └── 0_0.png          # Distance mask (black=land, white=water)
```

---

### Tile Formats

The web app supports two tile formats:

#### PMTiles (Recommended)

A single `.pmtiles` file containing all tiles and zoom levels:

```
src/app/public/
└── tiles.pmtiles        # Single archive with all tiles
```

#### Legacy Directory Structure

Individual PNG files organized by zoom level:

```
src/app/public/tiles/
├── 0/                    # Zoom level 0 = native resolution (max zoom in)
│   ├── 0_0.png          # Tile at x=0, y=0
│   ├── 0_1.png          # Tile at x=0, y=1
│   ├── 1_0.png          # Tile at x=1, y=0
│   └── ...
├── 1/                    # Zoom level 1 = 2× zoomed out
├── 2/                    # Zoom level 2 = 4× zoomed out
├── 3/                    # Zoom level 3 = 8× zoomed out
├── 4/                    # Zoom level 4 = 16× zoomed out
└── manifest.json         # Grid dimensions and metadata
```

**Tile naming:** `{zoom_level}/{x}_{y}.png`

**Zoom levels:**

| Level | Scale   | Description                 |
| ----- | ------- | --------------------------- |
| 0     | 1:1     | Native resolution (512×512) |
| 1     | 2× out  | 2×2 tiles combined          |
| 2     | 4× out  | 4×4 tiles combined          |
| 3     | 8× out  | 8×8 tiles combined          |
| 4     | 16× out | 16×16 tiles combined        |

---

## Python Pipeline

### Setup

Create a `.env` file in the root directory with API credentials:

```bash
GOOGLE_MAPS_API_KEY=...
NYC_OPENDATA_APP_TOKEN=...
```

Install dependencies using `uv`:

```bash
uv sync
```

### Scripts

```bash
# Generate tiles for an address
uv run isometric-nyc "350 5th Ave, New York, NY"

# Run the e2e generation viewer
uv run python src/isometric_nyc/e2e_generation/app.py

# Export tiles to the web app
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19
```

### Exporting Tiles for the Web App

There are two export formats available:

#### PMTiles Format (Recommended)

[PMTiles](https://github.com/protomaps/PMTiles) is a single-file archive format
for map tiles, optimized for cloud storage and CDN delivery. The web app
automatically detects and uses PMTiles when available.

```bash
# Export all tiles to a single .pmtiles file
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

# Export a specific region
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --tl 0,0 --br 49,49

# Custom output path
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 -o ./my-tiles.pmtiles

# Export without postprocessing (raw tiles)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --no-postprocess

# Customize postprocessing
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --scale 4 --colors 64 --no-dither

# Dry run to see what would be exported
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --dry-run
```

**Benefits of PMTiles:**

- **Single file**: All tiles and zoom levels in one `.pmtiles` file
- **Efficient serving**: Optimized for HTTP range requests from static storage
- **Smaller footprint**: Deduplicates identical tiles (e.g., black padding tiles)
- **Metadata included**: Grid dimensions, tile size, etc. embedded in the archive

The output file is placed at `src/app/public/tiles.pmtiles` by default.

#### Legacy Directory Format

The `export_tiles_for_app.py` script exports quadrants from the SQLite database
to individual PNG files in a directory structure:

```bash
# Export a 20x20 region, normalizing coordinates so (0,0) is top-left
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19

# Export a specific region (e.g., quadrants 10-29 → tiles 0-19)
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 10,10 --br 29,29

# Export render images instead of generations
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --render

# Custom output directory
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --output-dir ./my-tiles/0

# Dry run to see what would be exported
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --dry-run
```

**Coordinate normalization:** The `--tl` coordinate becomes `0_0.png` in the
output. For example, exporting `--tl 10,15 --br 20,25` produces tiles `0_0.png`
through `10_10.png`.

#### Web App Tile Loading

The web app automatically detects which format is available:

1. First tries to load `/tiles.pmtiles`
2. Falls back to `/tiles/manifest.json` + directory structure if PMTiles not found

This allows seamless migration to PMTiles while maintaining backward compatibility.

### Debug Map

The `debug_map.py` script generates a visualization showing generation progress
by overlaying generated quadrants on a real map of NYC.

```bash
# Generate debug map for the default generation directory
uv run python src/isometric_nyc/e2e_generation/debug_map.py

# Generate for a specific generation directory
uv run python src/isometric_nyc/e2e_generation/debug_map.py generations/v01

# Custom output path
uv run python src/isometric_nyc/e2e_generation/debug_map.py generations/v01 -o ./my_debug_map.png
```

**Features:**

- Overlays 25% alpha red rectangles for each generated quadrant
- Shows the seed point (green marker) as the generation origin
- Accounts for the isometric projection — quadrants appear as parallelograms
  matching the camera azimuth angle
- Uses CartoDB basemap tiles for geographic context
- Outputs to `debug_map.png` in the generation directory by default

**Output example:** Shows coverage gaps, generation progress, and geographic
extent of the current generation.

---

## Development

| Task                  | Command                                     |
| --------------------- | ------------------------------------------- |
| Run Python tests      | `uv run pytest`                             |
| Format Python code    | `uv run ruff format .`                      |
| Lint Python code      | `uv run ruff check .`                       |
| Run web app           | `cd src/app && bun run dev`                 |
| Build web app         | `cd src/app && bun run build`               |
| Run water shader demo | `cd src/water_shader_demo && bun dev`       |
| Build water shader    | `cd src/water_shader_demo && bun run build` |

---

## Deployment

The web app can be deployed to GitHub Pages with tile assets served from
Cloudflare R2.

### Prerequisites

1. **Install wrangler** (Cloudflare CLI):

   ```bash
   bun add -g wrangler
   ```

2. **Authenticate with Cloudflare**:

   ```bash
   wrangler login
   ```

3. **Create an R2 bucket** named `isometric-nyc` in your Cloudflare dashboard

### Deploy Script

The deployment script (`src/app/deploy.py`) handles three tasks:

| Step         | Flag             | Description                              |
| ------------ | ---------------- | ---------------------------------------- |
| Build        | `--build`        | Builds the Vite app to `dist/`           |
| GitHub Pages | `--github-pages` | Pushes HTML/JS/CSS to `gh-pages` branch  |
| Cloudflare R2| `--r2`           | Uploads PMTiles and tile assets to R2    |

### Usage

```bash
# Build only
uv run python src/app/deploy.py --build

# Deploy to GitHub Pages only
uv run python src/app/deploy.py --github-pages

# Upload tiles to R2 only
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
- **Cloudflare R2**: Serves tile assets and PMTiles (large files, CDN-optimized)

The web app automatically loads tiles from the configured R2 public URL.

```

Done! I've added a **Deployment** section to the README with:

- Prerequisites (wrangler installation and authentication)
- Deploy script usage with all flags
- Configuration options
- Architecture overview explaining the GitHub Pages + R2 split

---

**User**

Please try `uv run python src/app/deploy.py --all --dry-run` I'm getting lots of errors

---

**Cursor**

```plaintext
"build": "bun run tsc && bunx --bun vite build"
```

```json
"build": "bun run tsc && bunx --bun vite build",
```

The issue is that TypeScript isn't installed. Let me install dependencies first:

```json
"build": "bun run tsc && bunx --bun vite build",
```

Good, dependencies installed. But there's an issue with the build script - `bun run tsc` looks for a script, not the binary. Let me fix the package.json:

```json
{
  "name": "isometric-nyc-viewer",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "bunx --bun vite",
    "build": "bunx tsc && bunx --bun vite build",
    "preview": "bunx --bun vite preview",
    "generate-tiles": "bun run scripts/generate-placeholder-tiles.ts"
  },
  "dependencies": {
    "openseadragon": "^5.0.1",
    "pmtiles": "^4.3.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/bun": "^1.1.14",
    "@types/node": "^22.10.1",
    "@types/openseadragon": "^3.0.10",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "typescript": "^5.6.3",
    "vite": "^6.0.3"
  }
}

```

Now let's try the dry run again:

The dry run completed successfully! Here's what it found:

### GitHub Pages (small, fast-loading files)
- `index.html`
- `assets/index-*.js` and `assets/index-*.css`
- `palette.png`

### Cloudflare R2 (large tile assets)
- `tiles.pmtiles` — **5.1 GB** (!)
- 45,099 tile files

The issue was that `bun run tsc` was looking for a script, not the TypeScript binary. I fixed it to use `bunx tsc` instead.

**Note:** That `.DS_Store` file shouldn't be deployed. Want me to filter those out from the deploy script?

---

**User**

yes please

---

**Cursor**

```python
#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing pmtiles to Cloudflare R2 (using wrangler)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.parent
DIST_DIR = APP_DIR / "dist"
PUBLIC_DIR = APP_DIR / "public"

# GitHub Pages configuration
GITHUB_PAGES_BRANCH = "gh-pages"
GITHUB_PAGES_REMOTE = "origin"

# Cloudflare R2 configuration
R2_BUCKET_NAME = "isometric-nyc"

# Files to skip during deployment
SKIP_FILES = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}


def should_skip_file(file_path: Path) -> bool:
  """Check if a file should be skipped during deployment."""
  return file_path.name in SKIP_FILES or file_path.name.startswith("._")


def run_command(
  cmd: list[str],
  cwd: Path | None = None,
  check: bool = True,
  capture_output: bool = False,
) -> subprocess.CompletedProcess:
  """Run a shell command and handle errors."""
  print(f"  → Running: {' '.join(cmd)}")
  result = subprocess.run(
    cmd,
    cwd=cwd,
    check=check,
    capture_output=capture_output,
    text=True,
  )
  return result


def build_app() -> bool:
  """Build the static web app using bun/vite."""
  print("\n📦 Building web app...")

  # Check if bun is available
  if shutil.which("bun") is None:
    print("  ❌ Error: bun is not installed or not in PATH")
    print("  Install bun: curl -fsSL https://bun.sh/install | bash")
    return False

  # Install dependencies if needed
  if not (APP_DIR / "node_modules").exists():
    print("  📥 Installing dependencies...")
    run_command(["bun", "install"], cwd=APP_DIR)

  # Run the build
  print("  🔨 Running build...")
  run_command(["bun", "run", "build"], cwd=APP_DIR)

  if DIST_DIR.exists():
    print(f"  ✅ Build complete: {DIST_DIR}")
    return True
  else:
    print("  ❌ Build failed: dist directory not created")
    return False


def deploy_github_pages(dry_run: bool = False) -> bool:
  """Deploy built content to GitHub Pages branch."""
  print("\n🌐 Deploying to GitHub Pages...")

  if not DIST_DIR.exists():
    print("  ❌ Error: dist directory does not exist. Run build first.")
    return False

  # Create a temporary directory for the gh-pages content
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Copy dist contents to temp directory (excluding large assets)
    print("  📋 Preparing deployment files...")
    for item in DIST_DIR.iterdir():
      # Skip tiles and pmtiles - they'll be served from R2
      if (
        item.name in ("tiles", "tiles_processed", "water_masks")
        or item.suffix == ".pmtiles"
      ):
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      dest = tmp_path / item.name
      if item.is_dir():
        shutil.copytree(item, dest)
      else:
        shutil.copy2(item, dest)

    # Create .nojekyll file to disable Jekyll processing
    (tmp_path / ".nojekyll").touch()

    # Create CNAME file if a custom domain is configured
    custom_domain = os.environ.get("GITHUB_PAGES_DOMAIN")
    if custom_domain:
      (tmp_path / "CNAME").write_text(custom_domain)
      print(f"    📝 Created CNAME for {custom_domain}")

    if dry_run:
      print("  🔍 Dry run - would deploy these files:")
      for f in tmp_path.rglob("*"):
        if f.is_file():
          print(f"    - {f.relative_to(tmp_path)}")
      return True

    # Initialize git repo in temp directory
    print("  🔧 Initializing git repository...")
    run_command(["git", "init"], cwd=tmp_path)
    run_command(["git", "checkout", "-b", GITHUB_PAGES_BRANCH], cwd=tmp_path)

    # Get the remote URL from the main repo
    result = run_command(
      ["git", "remote", "get-url", GITHUB_PAGES_REMOTE],
      cwd=PROJECT_ROOT,
      capture_output=True,
    )
    remote_url = result.stdout.strip()

    run_command(["git", "remote", "add", GITHUB_PAGES_REMOTE, remote_url], cwd=tmp_path)

    # Add and commit all files
    run_command(["git", "add", "-A"], cwd=tmp_path)
    run_command(
      ["git", "commit", "-m", "Deploy to GitHub Pages"],
      cwd=tmp_path,
    )

    # Force push to gh-pages branch
    print("  🚀 Pushing to GitHub Pages...")
    run_command(
      ["git", "push", "--force", GITHUB_PAGES_REMOTE, GITHUB_PAGES_BRANCH],
      cwd=tmp_path,
    )

    print("  ✅ GitHub Pages deployment complete!")
    return True


def upload_to_r2(dry_run: bool = False) -> bool:
  """Upload pmtiles and tile assets to Cloudflare R2 using wrangler."""
  print("\n☁️  Uploading to Cloudflare R2...")

  # Check if wrangler is available
  if shutil.which("wrangler") is None:
    print("  ❌ Error: wrangler is not installed or not in PATH")
    print("  Install wrangler: bun add -g wrangler")
    print("  Then authenticate: wrangler login")
    return False

  # Find pmtiles file
  pmtiles_file = PUBLIC_DIR / "tiles.pmtiles"
  if not pmtiles_file.exists():
    pmtiles_file = DIST_DIR / "tiles.pmtiles"

  if not pmtiles_file.exists():
    print(f"  ⚠️  Warning: tiles.pmtiles not found in {PUBLIC_DIR} or {DIST_DIR}")
    pmtiles_file = None

  # Collect files to upload
  files_to_upload: list[tuple[Path, str]] = []

  if pmtiles_file:
    files_to_upload.append((pmtiles_file, "tiles.pmtiles"))

  # Add tile directories if they exist
  for tile_dir_name in ["tiles", "tiles_processed", "water_masks"]:
    tile_dir = PUBLIC_DIR / tile_dir_name
    if tile_dir.exists():
      for file_path in tile_dir.rglob("*"):
        if file_path.is_file():
          relative_path = file_path.relative_to(PUBLIC_DIR)
          files_to_upload.append((file_path, str(relative_path)))

  if not files_to_upload:
    print("  ⚠️  No files found to upload")
    return True

  print(f"  📊 Found {len(files_to_upload)} files to upload")

  if dry_run:
    print("  🔍 Dry run - would upload these files:")
    for local_path, remote_key in files_to_upload[:20]:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    - {remote_key} ({size_mb:.2f} MB)")
    if len(files_to_upload) > 20:
      print(f"    ... and {len(files_to_upload) - 20} more files")
    return True

  # Upload files using wrangler
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      print(f"    ⬆️  Uploading {remote_key}...", end=" ", flush=True)

      # Use wrangler r2 object put
      result = subprocess.run(
        [
          "wrangler",
          "r2",
          "object",
          "put",
          f"{R2_BUCKET_NAME}/{remote_key}",
          "--file",
          str(local_path),
        ],
        capture_output=True,
        text=True,
      )

      if result.returncode == 0:
        print("✓")
        uploaded += 1
      else:
        print(f"✗ ({result.stderr.strip()})")
        failed += 1

    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  print(f"\n  📊 Upload summary: {uploaded} succeeded, {failed} failed")

  if failed == 0:
    print("  ✅ R2 upload complete!")
    if pmtiles_file:
      print("  🔗 PMTiles will be available at your R2 public URL/tiles.pmtiles")
    return True
  else:
    print("  ⚠️  Some uploads failed")
    return False


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Prerequisites:
  - bun: Install with `curl -fsSL https://bun.sh/install | bash`
  - wrangler: Install with `bun add -g wrangler`, then run `wrangler login`

Environment Variables (optional):
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run

  # Just upload to R2
  python deploy.py --r2
    """,
  )

  parser.add_argument(
    "--build",
    action="store_true",
    help="Build the web app",
  )
  parser.add_argument(
    "--github-pages",
    action="store_true",
    help="Deploy to GitHub Pages",
  )
  parser.add_argument(
    "--r2",
    action="store_true",
    help="Upload to Cloudflare R2",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )

  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all:
    if not build_app():
      success = False
      if not args.all:
        return 1

  # GitHub Pages
  if args.github_pages or args.all:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  # R2 upload
  if args.r2 or args.all:
    if not upload_to_r2(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  print("\n" + "=" * 40)
  if success:
    print("✅ Deployment complete!")
  else:
    print("⚠️  Deployment completed with some errors")

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

```python
#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing pmtiles to Cloudflare R2 (using wrangler)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.parent
DIST_DIR = APP_DIR / "dist"
PUBLIC_DIR = APP_DIR / "public"

# GitHub Pages configuration
GITHUB_PAGES_BRANCH = "gh-pages"
GITHUB_PAGES_REMOTE = "origin"

# Cloudflare R2 configuration
R2_BUCKET_NAME = "isometric-nyc"

# Files to skip during deployment
SKIP_FILES = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}


def should_skip_file(file_path: Path) -> bool:
  """Check if a file should be skipped during deployment."""
  return file_path.name in SKIP_FILES or file_path.name.startswith("._")


def run_command(
  cmd: list[str],
  cwd: Path | None = None,
  check: bool = True,
  capture_output: bool = False,
) -> subprocess.CompletedProcess:
  """Run a shell command and handle errors."""
  print(f"  → Running: {' '.join(cmd)}")
  result = subprocess.run(
    cmd,
    cwd=cwd,
    check=check,
    capture_output=capture_output,
    text=True,
  )
  return result


def build_app() -> bool:
  """Build the static web app using bun/vite."""
  print("\n📦 Building web app...")

  # Check if bun is available
  if shutil.which("bun") is None:
    print("  ❌ Error: bun is not installed or not in PATH")
    print("  Install bun: curl -fsSL https://bun.sh/install | bash")
    return False

  # Install dependencies if needed
  if not (APP_DIR / "node_modules").exists():
    print("  📥 Installing dependencies...")
    run_command(["bun", "install"], cwd=APP_DIR)

  # Run the build
  print("  🔨 Running build...")
  run_command(["bun", "run", "build"], cwd=APP_DIR)

  if DIST_DIR.exists():
    print(f"  ✅ Build complete: {DIST_DIR}")
    return True
  else:
    print("  ❌ Build failed: dist directory not created")
    return False


def deploy_github_pages(dry_run: bool = False) -> bool:
  """Deploy built content to GitHub Pages branch."""
  print("\n🌐 Deploying to GitHub Pages...")

  if not DIST_DIR.exists():
    print("  ❌ Error: dist directory does not exist. Run build first.")
    return False

  # Create a temporary directory for the gh-pages content
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Copy dist contents to temp directory (excluding large assets)
    print("  📋 Preparing deployment files...")
    for item in DIST_DIR.iterdir():
      # Skip tiles and pmtiles - they'll be served from R2
      if (
        item.name in ("tiles", "tiles_processed", "water_masks")
        or item.suffix == ".pmtiles"
      ):
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      # Skip junk files
      if should_skip_file(item):
        continue

      dest = tmp_path / item.name
      if item.is_dir():
        shutil.copytree(
          item,
          dest,
          ignore=shutil.ignore_patterns(*SKIP_FILES, "._*"),
        )
      else:
        shutil.copy2(item, dest)

    # Create .nojekyll file to disable Jekyll processing
    (tmp_path / ".nojekyll").touch()

    # Create CNAME file if a custom domain is configured
    custom_domain = os.environ.get("GITHUB_PAGES_DOMAIN")
    if custom_domain:
      (tmp_path / "CNAME").write_text(custom_domain)
      print(f"    📝 Created CNAME for {custom_domain}")

    if dry_run:
      print("  🔍 Dry run - would deploy these files:")
      for f in tmp_path.rglob("*"):
        if f.is_file():
          print(f"    - {f.relative_to(tmp_path)}")
      return True

    # Initialize git repo in temp directory
    print("  🔧 Initializing git repository...")
    run_command(["git", "init"], cwd=tmp_path)
    run_command(["git", "checkout", "-b", GITHUB_PAGES_BRANCH], cwd=tmp_path)

    # Get the remote URL from the main repo
    result = run_command(
      ["git", "remote", "get-url", GITHUB_PAGES_REMOTE],
      cwd=PROJECT_ROOT,
      capture_output=True,
    )
    remote_url = result.stdout.strip()

    run_command(["git", "remote", "add", GITHUB_PAGES_REMOTE, remote_url], cwd=tmp_path)

    # Add and commit all files
    run_command(["git", "add", "-A"], cwd=tmp_path)
    run_command(
      ["git", "commit", "-m", "Deploy to GitHub Pages"],
      cwd=tmp_path,
    )

    # Force push to gh-pages branch
    print("  🚀 Pushing to GitHub Pages...")
    run_command(
      ["git", "push", "--force", GITHUB_PAGES_REMOTE, GITHUB_PAGES_BRANCH],
      cwd=tmp_path,
    )

    print("  ✅ GitHub Pages deployment complete!")
    return True


def upload_to_r2(dry_run: bool = False) -> bool:
  """Upload pmtiles and tile assets to Cloudflare R2 using wrangler."""
  print("\n☁️  Uploading to Cloudflare R2...")

  # Check if wrangler is available
  if shutil.which("wrangler") is None:
    print("  ❌ Error: wrangler is not installed or not in PATH")
    print("  Install wrangler: bun add -g wrangler")
    print("  Then authenticate: wrangler login")
    return False

  # Find pmtiles file
  pmtiles_file = PUBLIC_DIR / "tiles.pmtiles"
  if not pmtiles_file.exists():
    pmtiles_file = DIST_DIR / "tiles.pmtiles"

  if not pmtiles_file.exists():
    print(f"  ⚠️  Warning: tiles.pmtiles not found in {PUBLIC_DIR} or {DIST_DIR}")
    pmtiles_file = None

  # Collect files to upload
  files_to_upload: list[tuple[Path, str]] = []

  if pmtiles_file:
    files_to_upload.append((pmtiles_file, "tiles.pmtiles"))

  # Add tile directories if they exist
  for tile_dir_name in ["tiles", "tiles_processed", "water_masks"]:
    tile_dir = PUBLIC_DIR / tile_dir_name
    if tile_dir.exists():
      for file_path in tile_dir.rglob("*"):
        if file_path.is_file():
          relative_path = file_path.relative_to(PUBLIC_DIR)
          files_to_upload.append((file_path, str(relative_path)))

  if not files_to_upload:
    print("  ⚠️  No files found to upload")
    return True

  print(f"  📊 Found {len(files_to_upload)} files to upload")

  if dry_run:
    print("  🔍 Dry run - would upload these files:")
    for local_path, remote_key in files_to_upload[:20]:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    - {remote_key} ({size_mb:.2f} MB)")
    if len(files_to_upload) > 20:
      print(f"    ... and {len(files_to_upload) - 20} more files")
    return True

  # Upload files using wrangler
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      print(f"    ⬆️  Uploading {remote_key}...", end=" ", flush=True)

      # Use wrangler r2 object put
      result = subprocess.run(
        [
          "wrangler",
          "r2",
          "object",
          "put",
          f"{R2_BUCKET_NAME}/{remote_key}",
          "--file",
          str(local_path),
        ],
        capture_output=True,
        text=True,
      )

      if result.returncode == 0:
        print("✓")
        uploaded += 1
      else:
        print(f"✗ ({result.stderr.strip()})")
        failed += 1

    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  print(f"\n  📊 Upload summary: {uploaded} succeeded, {failed} failed")

  if failed == 0:
    print("  ✅ R2 upload complete!")
    if pmtiles_file:
      print("  🔗 PMTiles will be available at your R2 public URL/tiles.pmtiles")
    return True
  else:
    print("  ⚠️  Some uploads failed")
    return False


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Prerequisites:
  - bun: Install with `curl -fsSL https://bun.sh/install | bash`
  - wrangler: Install with `bun add -g wrangler`, then run `wrangler login`

Environment Variables (optional):
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run

  # Just upload to R2
  python deploy.py --r2
    """,
  )

  parser.add_argument(
    "--build",
    action="store_true",
    help="Build the web app",
  )
  parser.add_argument(
    "--github-pages",
    action="store_true",
    help="Deploy to GitHub Pages",
  )
  parser.add_argument(
    "--r2",
    action="store_true",
    help="Upload to Cloudflare R2",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )

  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all:
    if not build_app():
      success = False
      if not args.all:
        return 1

  # GitHub Pages
  if args.github_pages or args.all:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  # R2 upload
  if args.r2 or args.all:
    if not upload_to_r2(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  print("\n" + "=" * 40)
  if success:
    print("✅ Deployment complete!")
  else:
    print("⚠️  Deployment completed with some errors")

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

```python
#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing pmtiles to Cloudflare R2 (using wrangler)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.parent
DIST_DIR = APP_DIR / "dist"
PUBLIC_DIR = APP_DIR / "public"

# GitHub Pages configuration
GITHUB_PAGES_BRANCH = "gh-pages"
GITHUB_PAGES_REMOTE = "origin"

# Cloudflare R2 configuration
R2_BUCKET_NAME = "isometric-nyc"

# Files to skip during deployment
SKIP_FILES = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}


def should_skip_file(file_path: Path) -> bool:
  """Check if a file should be skipped during deployment."""
  return file_path.name in SKIP_FILES or file_path.name.startswith("._")


def run_command(
  cmd: list[str],
  cwd: Path | None = None,
  check: bool = True,
  capture_output: bool = False,
) -> subprocess.CompletedProcess:
  """Run a shell command and handle errors."""
  print(f"  → Running: {' '.join(cmd)}")
  result = subprocess.run(
    cmd,
    cwd=cwd,
    check=check,
    capture_output=capture_output,
    text=True,
  )
  return result


def build_app() -> bool:
  """Build the static web app using bun/vite."""
  print("\n📦 Building web app...")

  # Check if bun is available
  if shutil.which("bun") is None:
    print("  ❌ Error: bun is not installed or not in PATH")
    print("  Install bun: curl -fsSL https://bun.sh/install | bash")
    return False

  # Install dependencies if needed
  if not (APP_DIR / "node_modules").exists():
    print("  📥 Installing dependencies...")
    run_command(["bun", "install"], cwd=APP_DIR)

  # Run the build
  print("  🔨 Running build...")
  run_command(["bun", "run", "build"], cwd=APP_DIR)

  if DIST_DIR.exists():
    print(f"  ✅ Build complete: {DIST_DIR}")
    return True
  else:
    print("  ❌ Build failed: dist directory not created")
    return False


def deploy_github_pages(dry_run: bool = False) -> bool:
  """Deploy built content to GitHub Pages branch."""
  print("\n🌐 Deploying to GitHub Pages...")

  if not DIST_DIR.exists():
    print("  ❌ Error: dist directory does not exist. Run build first.")
    return False

  # Create a temporary directory for the gh-pages content
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Copy dist contents to temp directory (excluding large assets)
    print("  📋 Preparing deployment files...")
    for item in DIST_DIR.iterdir():
      # Skip tiles and pmtiles - they'll be served from R2
      if (
        item.name in ("tiles", "tiles_processed", "water_masks")
        or item.suffix == ".pmtiles"
      ):
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      # Skip junk files
      if should_skip_file(item):
        continue

      dest = tmp_path / item.name
      if item.is_dir():
        shutil.copytree(
          item,
          dest,
          ignore=shutil.ignore_patterns(*SKIP_FILES, "._*"),
        )
      else:
        shutil.copy2(item, dest)

    # Create .nojekyll file to disable Jekyll processing
    (tmp_path / ".nojekyll").touch()

    # Create CNAME file if a custom domain is configured
    custom_domain = os.environ.get("GITHUB_PAGES_DOMAIN")
    if custom_domain:
      (tmp_path / "CNAME").write_text(custom_domain)
      print(f"    📝 Created CNAME for {custom_domain}")

    if dry_run:
      print("  🔍 Dry run - would deploy these files:")
      for f in tmp_path.rglob("*"):
        if f.is_file():
          print(f"    - {f.relative_to(tmp_path)}")
      return True

    # Initialize git repo in temp directory
    print("  🔧 Initializing git repository...")
    run_command(["git", "init"], cwd=tmp_path)
    run_command(["git", "checkout", "-b", GITHUB_PAGES_BRANCH], cwd=tmp_path)

    # Get the remote URL from the main repo
    result = run_command(
      ["git", "remote", "get-url", GITHUB_PAGES_REMOTE],
      cwd=PROJECT_ROOT,
      capture_output=True,
    )
    remote_url = result.stdout.strip()

    run_command(["git", "remote", "add", GITHUB_PAGES_REMOTE, remote_url], cwd=tmp_path)

    # Add and commit all files
    run_command(["git", "add", "-A"], cwd=tmp_path)
    run_command(
      ["git", "commit", "-m", "Deploy to GitHub Pages"],
      cwd=tmp_path,
    )

    # Force push to gh-pages branch
    print("  🚀 Pushing to GitHub Pages...")
    run_command(
      ["git", "push", "--force", GITHUB_PAGES_REMOTE, GITHUB_PAGES_BRANCH],
      cwd=tmp_path,
    )

    print("  ✅ GitHub Pages deployment complete!")
    return True


def upload_to_r2(dry_run: bool = False) -> bool:
  """Upload pmtiles and tile assets to Cloudflare R2 using wrangler."""
  print("\n☁️  Uploading to Cloudflare R2...")

  # Check if wrangler is available
  if shutil.which("wrangler") is None:
    print("  ❌ Error: wrangler is not installed or not in PATH")
    print("  Install wrangler: bun add -g wrangler")
    print("  Then authenticate: wrangler login")
    return False

  # Find pmtiles file
  pmtiles_file = PUBLIC_DIR / "tiles.pmtiles"
  if not pmtiles_file.exists():
    pmtiles_file = DIST_DIR / "tiles.pmtiles"

  if not pmtiles_file.exists():
    print(f"  ⚠️  Warning: tiles.pmtiles not found in {PUBLIC_DIR} or {DIST_DIR}")
    pmtiles_file = None

  # Collect files to upload
  files_to_upload: list[tuple[Path, str]] = []

  if pmtiles_file:
    files_to_upload.append((pmtiles_file, "tiles.pmtiles"))

  # Add tile directories if they exist
  for tile_dir_name in ["tiles", "tiles_processed", "water_masks"]:
    tile_dir = PUBLIC_DIR / tile_dir_name
    if tile_dir.exists():
      for file_path in tile_dir.rglob("*"):
        if file_path.is_file() and not should_skip_file(file_path):
          relative_path = file_path.relative_to(PUBLIC_DIR)
          files_to_upload.append((file_path, str(relative_path)))

  if not files_to_upload:
    print("  ⚠️  No files found to upload")
    return True

  print(f"  📊 Found {len(files_to_upload)} files to upload")

  if dry_run:
    print("  🔍 Dry run - would upload these files:")
    for local_path, remote_key in files_to_upload[:20]:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    - {remote_key} ({size_mb:.2f} MB)")
    if len(files_to_upload) > 20:
      print(f"    ... and {len(files_to_upload) - 20} more files")
    return True

  # Upload files using wrangler
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      print(f"    ⬆️  Uploading {remote_key}...", end=" ", flush=True)

      # Use wrangler r2 object put
      result = subprocess.run(
        [
          "wrangler",
          "r2",
          "object",
          "put",
          f"{R2_BUCKET_NAME}/{remote_key}",
          "--file",
          str(local_path),
        ],
        capture_output=True,
        text=True,
      )

      if result.returncode == 0:
        print("✓")
        uploaded += 1
      else:
        print(f"✗ ({result.stderr.strip()})")
        failed += 1

    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  print(f"\n  📊 Upload summary: {uploaded} succeeded, {failed} failed")

  if failed == 0:
    print("  ✅ R2 upload complete!")
    if pmtiles_file:
      print("  🔗 PMTiles will be available at your R2 public URL/tiles.pmtiles")
    return True
  else:
    print("  ⚠️  Some uploads failed")
    return False


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Prerequisites:
  - bun: Install with `curl -fsSL https://bun.sh/install | bash`
  - wrangler: Install with `bun add -g wrangler`, then run `wrangler login`

Environment Variables (optional):
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run

  # Just upload to R2
  python deploy.py --r2
    """,
  )

  parser.add_argument(
    "--build",
    action="store_true",
    help="Build the web app",
  )
  parser.add_argument(
    "--github-pages",
    action="store_true",
    help="Deploy to GitHub Pages",
  )
  parser.add_argument(
    "--r2",
    action="store_true",
    help="Upload to Cloudflare R2",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )

  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all:
    if not build_app():
      success = False
      if not args.all:
        return 1

  # GitHub Pages
  if args.github_pages or args.all:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  # R2 upload
  if args.r2 or args.all:
    if not upload_to_r2(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  print("\n" + "=" * 40)
  if success:
    print("✅ Deployment complete!")
  else:
    print("⚠️  Deployment completed with some errors")

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

Done! The `.DS_Store` files are now filtered out:

- **GitHub Pages**: No more `.DS_Store` in the deploy list
- **R2**: Went from 45,099 → 45,096 files (3 `.DS_Store` files removed)

The script now skips `.DS_Store`, `Thumbs.db`, `.gitkeep`, `.gitignore`, and any `._*` files.

---

**User**

Please don't upload the pngs - only the .pmtiles files

---

**Cursor**

```python
#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing pmtiles to Cloudflare R2 (using wrangler)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.parent
DIST_DIR = APP_DIR / "dist"
PUBLIC_DIR = APP_DIR / "public"

# GitHub Pages configuration
GITHUB_PAGES_BRANCH = "gh-pages"
GITHUB_PAGES_REMOTE = "origin"

# Cloudflare R2 configuration
R2_BUCKET_NAME = "isometric-nyc"

# Files to skip during deployment
SKIP_FILES = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}


def should_skip_file(file_path: Path) -> bool:
  """Check if a file should be skipped during deployment."""
  return file_path.name in SKIP_FILES or file_path.name.startswith("._")


def run_command(
  cmd: list[str],
  cwd: Path | None = None,
  check: bool = True,
  capture_output: bool = False,
) -> subprocess.CompletedProcess:
  """Run a shell command and handle errors."""
  print(f"  → Running: {' '.join(cmd)}")
  result = subprocess.run(
    cmd,
    cwd=cwd,
    check=check,
    capture_output=capture_output,
    text=True,
  )
  return result


def build_app() -> bool:
  """Build the static web app using bun/vite."""
  print("\n📦 Building web app...")

  # Check if bun is available
  if shutil.which("bun") is None:
    print("  ❌ Error: bun is not installed or not in PATH")
    print("  Install bun: curl -fsSL https://bun.sh/install | bash")
    return False

  # Install dependencies if needed
  if not (APP_DIR / "node_modules").exists():
    print("  📥 Installing dependencies...")
    run_command(["bun", "install"], cwd=APP_DIR)

  # Run the build
  print("  🔨 Running build...")
  run_command(["bun", "run", "build"], cwd=APP_DIR)

  if DIST_DIR.exists():
    print(f"  ✅ Build complete: {DIST_DIR}")
    return True
  else:
    print("  ❌ Build failed: dist directory not created")
    return False


def deploy_github_pages(dry_run: bool = False) -> bool:
  """Deploy built content to GitHub Pages branch."""
  print("\n🌐 Deploying to GitHub Pages...")

  if not DIST_DIR.exists():
    print("  ❌ Error: dist directory does not exist. Run build first.")
    return False

  # Create a temporary directory for the gh-pages content
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Copy dist contents to temp directory (excluding large assets)
    print("  📋 Preparing deployment files...")
    for item in DIST_DIR.iterdir():
      # Skip tiles and pmtiles - they'll be served from R2
      if (
        item.name in ("tiles", "tiles_processed", "water_masks")
        or item.suffix == ".pmtiles"
      ):
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      # Skip junk files
      if should_skip_file(item):
        continue

      dest = tmp_path / item.name
      if item.is_dir():
        shutil.copytree(
          item,
          dest,
          ignore=shutil.ignore_patterns(*SKIP_FILES, "._*"),
        )
      else:
        shutil.copy2(item, dest)

    # Create .nojekyll file to disable Jekyll processing
    (tmp_path / ".nojekyll").touch()

    # Create CNAME file if a custom domain is configured
    custom_domain = os.environ.get("GITHUB_PAGES_DOMAIN")
    if custom_domain:
      (tmp_path / "CNAME").write_text(custom_domain)
      print(f"    📝 Created CNAME for {custom_domain}")

    if dry_run:
      print("  🔍 Dry run - would deploy these files:")
      for f in tmp_path.rglob("*"):
        if f.is_file():
          print(f"    - {f.relative_to(tmp_path)}")
      return True

    # Initialize git repo in temp directory
    print("  🔧 Initializing git repository...")
    run_command(["git", "init"], cwd=tmp_path)
    run_command(["git", "checkout", "-b", GITHUB_PAGES_BRANCH], cwd=tmp_path)

    # Get the remote URL from the main repo
    result = run_command(
      ["git", "remote", "get-url", GITHUB_PAGES_REMOTE],
      cwd=PROJECT_ROOT,
      capture_output=True,
    )
    remote_url = result.stdout.strip()

    run_command(["git", "remote", "add", GITHUB_PAGES_REMOTE, remote_url], cwd=tmp_path)

    # Add and commit all files
    run_command(["git", "add", "-A"], cwd=tmp_path)
    run_command(
      ["git", "commit", "-m", "Deploy to GitHub Pages"],
      cwd=tmp_path,
    )

    # Force push to gh-pages branch
    print("  🚀 Pushing to GitHub Pages...")
    run_command(
      ["git", "push", "--force", GITHUB_PAGES_REMOTE, GITHUB_PAGES_BRANCH],
      cwd=tmp_path,
    )

    print("  ✅ GitHub Pages deployment complete!")
    return True


def upload_to_r2(dry_run: bool = False) -> bool:
  """Upload pmtiles and tile assets to Cloudflare R2 using wrangler."""
  print("\n☁️  Uploading to Cloudflare R2...")

  # Check if wrangler is available
  if shutil.which("wrangler") is None:
    print("  ❌ Error: wrangler is not installed or not in PATH")
    print("  Install wrangler: bun add -g wrangler")
    print("  Then authenticate: wrangler login")
    return False

  # Find pmtiles file
  pmtiles_file = PUBLIC_DIR / "tiles.pmtiles"
  if not pmtiles_file.exists():
    pmtiles_file = DIST_DIR / "tiles.pmtiles"

  if not pmtiles_file.exists():
    print(f"  ⚠️  Warning: tiles.pmtiles not found in {PUBLIC_DIR} or {DIST_DIR}")
    pmtiles_file = None

  # Collect files to upload (only pmtiles)
  files_to_upload: list[tuple[Path, str]] = []

  if pmtiles_file:
    files_to_upload.append((pmtiles_file, "tiles.pmtiles"))

  if not files_to_upload:
    print("  ⚠️  No files found to upload")
    return True

  print(f"  📊 Found {len(files_to_upload)} files to upload")

  if dry_run:
    print("  🔍 Dry run - would upload these files:")
    for local_path, remote_key in files_to_upload[:20]:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    - {remote_key} ({size_mb:.2f} MB)")
    if len(files_to_upload) > 20:
      print(f"    ... and {len(files_to_upload) - 20} more files")
    return True

  # Upload files using wrangler
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      print(f"    ⬆️  Uploading {remote_key}...", end=" ", flush=True)

      # Use wrangler r2 object put
      result = subprocess.run(
        [
          "wrangler",
          "r2",
          "object",
          "put",
          f"{R2_BUCKET_NAME}/{remote_key}",
          "--file",
          str(local_path),
        ],
        capture_output=True,
        text=True,
      )

      if result.returncode == 0:
        print("✓")
        uploaded += 1
      else:
        print(f"✗ ({result.stderr.strip()})")
        failed += 1

    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  print(f"\n  📊 Upload summary: {uploaded} succeeded, {failed} failed")

  if failed == 0:
    print("  ✅ R2 upload complete!")
    if pmtiles_file:
      print("  🔗 PMTiles will be available at your R2 public URL/tiles.pmtiles")
    return True
  else:
    print("  ⚠️  Some uploads failed")
    return False


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Prerequisites:
  - bun: Install with `curl -fsSL https://bun.sh/install | bash`
  - wrangler: Install with `bun add -g wrangler`, then run `wrangler login`

Environment Variables (optional):
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run

  # Just upload to R2
  python deploy.py --r2
    """,
  )

  parser.add_argument(
    "--build",
    action="store_true",
    help="Build the web app",
  )
  parser.add_argument(
    "--github-pages",
    action="store_true",
    help="Deploy to GitHub Pages",
  )
  parser.add_argument(
    "--r2",
    action="store_true",
    help="Upload to Cloudflare R2",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )

  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all:
    if not build_app():
      success = False
      if not args.all:
        return 1

  # GitHub Pages
  if args.github_pages or args.all:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  # R2 upload
  if args.r2 or args.all:
    if not upload_to_r2(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  print("\n" + "=" * 40)
  if success:
    print("✅ Deployment complete!")
  else:
    print("⚠️  Deployment completed with some errors")

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

Done! Now R2 upload only includes the `.pmtiles` file (5.1 GB), not the 45k+ individual PNGs.

---

**User**

I sthere any way to compress the pmtiles that we export? 5 GB is huge!!!!

---

**Cursor**

Let me check the PMTiles export script to see what compression options are available:

The 5 GB file is huge because you have 128×128 = 16,384 base tiles + zoomed out versions. Here are your options:

### Option 1: Switch to WebP (Recommended - 25-35% smaller)
WebP is smaller than PNG and has great browser support. I can update the export script to use WebP instead.

### Option 2: More aggressive PNG compression
Use `pngquant` or `oxipng` to compress PNGs better before writing to PMTiles.

### Option 3: Increase pixel scale
Current: `--scale 2` → Try `--scale 4` for blockier pixels (smaller file but less detail)

### Option 4: Reduce color palette
Current: `--colors 32` → Try `--colors 16` for fewer colors

### Quick test - what's the current export command?
You can try exporting with more aggressive settings:
```bash
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --scale 4 --colors 16
```

**Would you like me to add WebP support to the export script?** That's likely the biggest win for file size while maintaining quality.

```shellscript
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --scale 4 --colors 16
```

---

**User**

yeah, let's first add webp support

---

**Cursor**

```python
from pmtiles.tile import Compression, TileType, zxy_to_tileid
```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_png_bytes(black_tile)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_png_bytes(processed)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      return data
  else:
    return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to PNG bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.

  Returns:
      Dict mapping (x, y) to PNG bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_png_bytes(combined.convert("RGB"))

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(palette_img, pixel_scale, dither)

  # First, load all base (level 0) tiles into memory
  print(f"\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles, padded_width, padded_height, level, black_tile_bytes
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})")
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants...")
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding")
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})")
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())


```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_png_bytes(black_tile)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_png_bytes(processed)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      return data
  else:
    return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to PNG bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.

  Returns:
      Dict mapping (x, y) to PNG bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_png_bytes(combined.convert("RGB"))

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(palette_img, pixel_scale, dither)

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles, padded_width, padded_height, level, black_tile_bytes
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_png_bytes(processed)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      return data
  else:
    return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to PNG bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.

  Returns:
      Dict mapping (x, y) to PNG bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_png_bytes(combined.convert("RGB"))

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(palette_img, pixel_scale, dither)

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles, padded_width, padded_height, level, black_tile_bytes
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to PNG bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.

  Returns:
      Dict mapping (x, y) to PNG bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_png_bytes(combined.convert("RGB"))

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(palette_img, pixel_scale, dither)

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles, padded_width, padded_height, level, black_tile_bytes
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(combined.convert("RGB"), image_format, webp_quality)

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(palette_img, pixel_scale, dither)

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles, padded_width, padded_height, level, black_tile_bytes
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(combined.convert("RGB"), image_format, webp_quality)

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles, padded_width, padded_height, level, black_tile_bytes
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(combined.convert("RGB"), image_format, webp_quality)

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles, padded_width, padded_height, level, black_tile_bytes
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  # Determine image format
  image_format = FORMAT_WEBP if args.webp else FORMAT_PNG
  print(f"🖼️  Image format: {image_format.upper()}")
  if args.webp:
    print(f"   WebP quality: {args.webp_quality}")
  print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Format: {image_format.upper()}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  # Determine image format
  image_format = FORMAT_WEBP if args.webp else FORMAT_PNG
  print(f"🖼️  Image format: {image_format.upper()}")
  if args.webp:
    print(f"   WebP quality: {args.webp_quality}")
  print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Format: {image_format.upper()}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
    image_format=image_format,
    webp_quality=args.webp_quality,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  file_size_mb = output_path.stat().st_size / 1024 / 1024
  file_size_gb = file_size_mb / 1024
  if file_size_gb >= 1:
    print(f"   File size: {file_size_gb:.2f} GB")
  else:
    print(f"   File size: {file_size_mb:.2f} MB")
  print(f"   Format: {image_format.upper()}")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding")
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})")
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  # Determine image format
  image_format = FORMAT_WEBP if args.webp else FORMAT_PNG
  print(f"🖼️  Image format: {image_format.upper()}")
  if args.webp:
    print(f"   WebP quality: {args.webp_quality}")
  print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Format: {image_format.upper()}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
    image_format=image_format,
    webp_quality=args.webp_quality,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  file_size_mb = output_path.stat().st_size / 1024 / 1024
  file_size_gb = file_size_mb / 1024
  if file_size_gb >= 1:
    print(f"   File size: {file_size_gb:.2f} GB")
  else:
    print(f"   File size: {file_size_mb:.2f} MB")
  print(f"   Format: {image_format.upper()}")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Image formats:
  - PNG (default): Lossless, larger files
  - WebP (--webp): Lossy, typically 25-35% smaller files

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with WebP format (smaller files)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  # Determine image format
  image_format = FORMAT_WEBP if args.webp else FORMAT_PNG
  print(f"🖼️  Image format: {image_format.upper()}")
  if args.webp:
    print(f"   WebP quality: {args.webp_quality}")
  print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Format: {image_format.upper()}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
    image_format=image_format,
    webp_quality=args.webp_quality,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  file_size_mb = output_path.stat().st_size / 1024 / 1024
  file_size_gb = file_size_mb / 1024
  if file_size_gb >= 1:
    print(f"   File size: {file_size_gb:.2f} GB")
  else:
    print(f"   File size: {file_size_mb:.2f} MB")
  print(f"   Format: {image_format.upper()}")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

Done! WebP support has been added to the PMTiles export. To use it:

```bash
# Export with WebP (typically 25-35% smaller than PNG)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

# Adjust quality for even smaller files (lower = smaller but more artifacts)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp --webp-quality 75
```

**Expected file size reduction:**
- PNG @ 5.1 GB → WebP should be ~3-4 GB
- With `--webp-quality 75` → potentially ~2.5-3 GB

The web app doesn't need changes - browsers natively support WebP, and PMTiles handles the format automatically.

```shellscript
# Export with WebP (typically 25-35% smaller than PNG)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

# Adjust quality for even smaller files (lower = smaller but more artifacts)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp --webp-quality 75
```

---

**User**

OK next, for the deployed app I don't want the controls panel to show and I don't want the x,y tile hover menu to show

---

**Cursor**

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = "/tiles.pmtiles";

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch("/tiles/manifest.json")
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `/tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      <ControlPanel
        scanlines={scanlines}
        onScanlinesChange={setScanlines}
        waterShader={waterShader}
        onWaterShaderChange={setWaterShader}
      />

      <TileInfo hoveredTile={hoveredTile} viewState={viewState} />
    </div>
  );
}

export default App;

```

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = "/tiles.pmtiles";

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch("/tiles/manifest.json")
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `/tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {isDev && (
        <ControlPanel
          scanlines={scanlines}
          onScanlinesChange={setScanlines}
          waterShader={waterShader}
          onWaterShaderChange={setWaterShader}
        />
      )}

      {isDev && <TileInfo hoveredTile={hoveredTile} viewState={viewState} />}
    </div>
  );
}

export default App;

```

Done! The `ControlPanel` and `TileInfo` components will now only show in development mode (`bun run dev`). When you build and deploy (`bun run build`), they will be hidden.

The check uses Vite's built-in `import.meta.env.DEV` which is `true` during development and `false` in production builds.

---

**User**

Is there a way to export a pmtiles that's significantly smaller for debugging the deployment? Like generate a 16x16 or 32x32 quadrant only?

---

**Cursor**

Yes! The export script already supports `--tl` and `--br` flags to specify a region. Just export a small area:

```bash
# Export just a 16x16 region (starting from 0,0)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --tl 0,0 --br 15,15 --webp

# Or a 32x32 region
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --tl 0,0 --br 31,31 --webp

# Pick a specific area that has generated content (check your db bounds)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --tl 50,50 --br 65,65 --webp
```

The script will auto-detect the database bounds first, so you can see where your content is and pick an interesting region.

A 16×16 grid should be ~50-100 MB instead of 5 GB — much faster to test uploads!

```shellscript
# Export just a 16x16 region (starting from 0,0)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --tl 0,0 --br 15,15 --webp

# Or a 32x32 region
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --tl 0,0 --br 31,31 --webp

# Pick a specific area that has generated content (check your db bounds)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py \
  generations/v01 --tl 50,50 --br 65,65 --webp
```

---

**User**

OK - change the behavior of the export_pmtiles.py script to have default scale 1, default colors 256, and defaulting to NOT dithering

---

**Cursor**

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Image formats:
  - PNG (default): Lossless, larger files
  - WebP (--webp): Lossy, typically 25-35% smaller files

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with WebP format (smaller files)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 1
DEFAULT_NUM_COLORS = 256
DEFAULT_DITHER = False
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  # Determine image format
  image_format = FORMAT_WEBP if args.webp else FORMAT_PNG
  print(f"🖼️  Image format: {image_format.upper()}")
  if args.webp:
    print(f"   WebP quality: {args.webp_quality}")
  print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Format: {image_format.upper()}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
    image_format=image_format,
    webp_quality=args.webp_quality,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  file_size_mb = output_path.stat().st_size / 1024 / 1024
  file_size_gb = file_size_mb / 1024
  if file_size_gb >= 1:
    print(f"   File size: {file_size_gb:.2f} GB")
  else:
    print(f"   File size: {file_size_mb:.2f} MB")
  print(f"   Format: {image_format.upper()}")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Image formats:
  - PNG (default): Lossless, larger files
  - WebP (--webp): Lossy, typically 25-35% smaller files

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with WebP format (smaller files)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 1
DEFAULT_NUM_COLORS = 256
DEFAULT_DITHER = False
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--dither",
    action="store_true",
    help="Enable dithering (disabled by default for cleaner pixel art)",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  # Determine image format
  image_format = FORMAT_WEBP if args.webp else FORMAT_PNG
  print(f"🖼️  Image format: {image_format.upper()}")
  if args.webp:
    print(f"   WebP quality: {args.webp_quality}")
  print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Format: {image_format.upper()}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
    image_format=image_format,
    webp_quality=args.webp_quality,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  file_size_mb = output_path.stat().st_size / 1024 / 1024
  file_size_gb = file_size_mb / 1024
  if file_size_gb >= 1:
    print(f"   File size: {file_size_gb:.2f} GB")
  else:
    print(f"   File size: {file_size_mb:.2f} MB")
  print(f"   Format: {image_format.upper()}")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Image formats:
  - PNG (default): Lossless, larger files
  - WebP (--webp): Lossy, typically 25-35% smaller files

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with WebP format (smaller files)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 1
DEFAULT_NUM_COLORS = 256
DEFAULT_DITHER = False
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--dither",
    action="store_true",
    help="Enable dithering (disabled by default for cleaner pixel art)",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={args.dither}"
    )
    print()

  # Determine image format
  image_format = FORMAT_WEBP if args.webp else FORMAT_PNG
  print(f"🖼️  Image format: {image_format.upper()}")
  if args.webp:
    print(f"   WebP quality: {args.webp_quality}")
  print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Format: {image_format.upper()}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
    image_format=image_format,
    webp_quality=args.webp_quality,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  file_size_mb = output_path.stat().st_size / 1024 / 1024
  file_size_gb = file_size_mb / 1024
  if file_size_gb >= 1:
    print(f"   File size: {file_size_gb:.2f} GB")
  else:
    print(f"   File size: {file_size_mb:.2f} MB")
  print(f"   Format: {image_format.upper()}")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

```python
"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Image formats:
  - PNG (default): Lossless, larger files
  - WebP (--webp): Lossy, typically 25-35% smaller files

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with WebP format (smaller files)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
"""

import argparse
import io
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid

# Image format options
FORMAT_PNG = "png"
FORMAT_WEBP = "webp"
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
from pmtiles.writer import write as pmtiles_write

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 1
DEFAULT_NUM_COLORS = 256
DEFAULT_DITHER = False
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


def parse_coordinate(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '10,15' into (x, y) tuple."""
  try:
    parts = coord_str.split(",")
    if len(parts) != 2:
      raise ValueError()
    return int(parts[0].strip()), int(parts[1].strip())
  except (ValueError, IndexError):
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
    )


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """Get the image bytes for a quadrant at position (x, y)."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
  """Get the bounding box of all quadrants in the database."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
      FROM quadrants
      """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
      return row[0], row[1], row[2], row[3]
    return None
  finally:
    conn.close()


def count_generated_quadrants(
  db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT COUNT(*) FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    with_data = cursor.fetchone()[0]
    total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
    return total, with_data
  finally:
    conn.close()


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    cursor.execute(
      f"""
      SELECT quadrant_x, quadrant_y FROM quadrants
      WHERE quadrant_x >= ? AND quadrant_x <= ?
        AND quadrant_y >= ? AND quadrant_y <= ?
        AND {column} IS NOT NULL
      """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_bytes(
  img: Image.Image,
  format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Convert a PIL Image to PNG or WebP bytes."""
  buffer = io.BytesIO()
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_bytes(black_tile, image_format, webp_quality)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_bytes(processed, image_format, webp_quality)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      # Re-encode in target format even without postprocessing
      try:
        img = Image.open(io.BytesIO(data))
        return image_to_bytes(img, image_format, webp_quality)
      except Exception:
        return data
  else:
    # Re-encode in target format
    try:
      img = Image.open(io.BytesIO(data))
      return image_to_bytes(img, image_format, webp_quality)
    except Exception:
      return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to image bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.
      image_format: Output format (png or webp).
      webp_quality: Quality for WebP compression.

  Returns:
      Dict mapping (x, y) to image bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_bytes(
        combined.convert("RGB"), image_format, webp_quality
      )

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
  image_format: str = FORMAT_PNG,
  webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(
    palette_img, pixel_scale, dither, image_format, webp_quality
  )

  # First, load all base (level 0) tiles into memory
  print("\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
        image_format,
        webp_quality,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles,
      padded_width,
      padded_height,
      level,
      black_tile_bytes,
      image_format,
      webp_quality,
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    tile_type = TileType.WEBP if image_format == FORMAT_WEBP else TileType.PNG
    header = {
      "tile_type": tile_type,
      "tile_compression": Compression.NONE,  # Images are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": image_format,
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
    """,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--br",
    type=parse_coordinate,
    required=False,
    default=None,
    metavar="X,Y",
    help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render images instead of generation images",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be exported without actually exporting",
  )

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--dither",
    action="store_true",
    help="Enable dithering (disabled by default for cleaner pixel art)",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )

  # Image format arguments
  format_group = parser.add_argument_group("image format options")
  format_group.add_argument(
    "--webp",
    action="store_true",
    help="Use WebP format instead of PNG (typically 25-35%% smaller files)",
  )
  format_group.add_argument(
    "--webp-quality",
    type=int,
    default=DEFAULT_WEBP_QUALITY,
    help=f"WebP quality (0-100, default: {DEFAULT_WEBP_QUALITY}). Lower = smaller but more artifacts",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

  # Validate inputs
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get database bounds
  bounds = get_quadrant_bounds(db_path)
  if not bounds:
    print("❌ Error: No quadrants found in database")
    return 1

  print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

  # Use provided coordinates or auto-detect
  if args.tl is None or args.br is None:
    if args.tl is not None or args.br is not None:
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={args.dither}"
    )
    print()

  # Determine image format
  image_format = FORMAT_WEBP if args.webp else FORMAT_PNG
  print(f"🖼️  Image format: {image_format.upper()}")
  if args.webp:
    print(f"   WebP quality: {args.webp_quality}")
  print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Format: {image_format.upper()}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=args.dither,
    max_zoom=MAX_ZOOM_LEVEL,
    image_format=image_format,
    webp_quality=args.webp_quality,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  file_size_mb = output_path.stat().st_size / 1024 / 1024
  file_size_gb = file_size_mb / 1024
  if file_size_gb >= 1:
    print(f"   File size: {file_size_gb:.2f} GB")
  else:
    print(f"   File size: {file_size_mb:.2f} MB")
  print(f"   Format: {image_format.upper()}")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(
    f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding"
  )
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

```

Done! The new defaults are:

| Setting | Old Default | New Default |
|---------|------------|-------------|
| Scale | 2 | 1 (no pixelation) |
| Colors | 32 | 256 |
| Dither | enabled | disabled |

The flag changed from `--no-dither` to `--dither` (now opt-in instead of opt-out).

---

**User**

what's the command to deploy?

---

**Cursor**

```bash
# Full deployment (build + GitHub Pages + R2)
uv run python src/app/deploy.py --all

# Or step by step:
uv run python src/app/deploy.py --build          # Build only
uv run python src/app/deploy.py --github-pages   # Push to gh-pages branch
uv run python src/app/deploy.py --r2             # Upload pmtiles to R2

# Dry run to preview without changes
uv run python src/app/deploy.py --all --dry-run
```

```shellscript
# Full deployment (build + GitHub Pages + R2)
uv run python src/app/deploy.py --all

# Or step by step:
uv run python src/app/deploy.py --build          # Build only
uv run python src/app/deploy.py --github-pages   # Push to gh-pages branch
uv run python src/app/deploy.py --r2             # Upload pmtiles to R2

# Dry run to preview without changes
uv run python src/app/deploy.py --all --dry-run
```

---

**User**

where do i get the R2 access key id and access key?

---

**Cursor**

Good news - you **don't need API keys**! The deploy script uses `wrangler` which authenticates via browser login.

Just run:
```bash
wrangler login
```

This opens a browser to authenticate with your Cloudflare account. Once logged in, wrangler stores the credentials locally and the deploy script will work automatically.

---

If you ever need R2 API keys (for other tools), you'd find them at:
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Select your account → **R2 Object Storage**
3. Click **Manage R2 API Tokens**
4. Create a token with read/write access to your bucket

But for this deploy script, `wrangler login` is all you need!

```shellscript
wrangler login
```

---

**User**

OK - we need to update the root of the html/js/css assets to be "cannoneyed.com/isometric-nyc", that's the domain that github pages will point to

---

**Cursor**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => ({
  // Use /isometric-nyc/ base path for production (GitHub Pages)
  base: mode === "production" ? "/isometric-nyc/" : "/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  server: {
    port: 3000,
    open: true,
  },
}));

```

The asset URLs in the code also need to use the base path. Let me update them to use Vite's `BASE_URL`:

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = `${import.meta.env.BASE_URL}tiles.pmtiles`;

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch("/tiles/manifest.json")
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `/tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {isDev && (
        <ControlPanel
          scanlines={scanlines}
          onScanlinesChange={setScanlines}
          waterShader={waterShader}
          onWaterShaderChange={setWaterShader}
        />
      )}

      {isDev && <TileInfo hoveredTile={hoveredTile} viewState={viewState} />}
    </div>
  );
}

export default App;

```

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = `${import.meta.env.BASE_URL}tiles.pmtiles`;

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch(`${import.meta.env.BASE_URL}tiles/manifest.json`)
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `/tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {isDev && (
        <ControlPanel
          scanlines={scanlines}
          onScanlinesChange={setScanlines}
          waterShader={waterShader}
          onWaterShaderChange={setWaterShader}
        />
      )}

      {isDev && <TileInfo hoveredTile={hoveredTile} viewState={viewState} />}
    </div>
  );
}

export default App;

```

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = `${import.meta.env.BASE_URL}tiles.pmtiles`;

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch(`${import.meta.env.BASE_URL}tiles/manifest.json`)
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `${import.meta.env.BASE_URL}tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {isDev && (
        <ControlPanel
          scanlines={scanlines}
          onScanlinesChange={setScanlines}
          waterShader={waterShader}
          onWaterShaderChange={setWaterShader}
        />
      )}

      {isDev && <TileInfo hoveredTile={hoveredTile} viewState={viewState} />}
    </div>
  );
}

export default App;

```

```typescriptreact
import { useEffect, useRef, useCallback } from "react";
import OpenSeadragon from "openseadragon";
import { PMTiles } from "pmtiles";
import type { ViewState } from "../App";
import { ScanlineOverlay } from "./ScanlineOverlay";
import { WaterShaderOverlay } from "./WaterShaderOverlay";
import type { ShaderParams } from "../shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file (optional, falls back to tile directory)
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

interface ScanlineSettings {
  enabled: boolean;
  count: number;
  opacity: number;
}

interface WaterShaderSettings {
  enabled: boolean;
  showMask: boolean;
  params: ShaderParams;
}

interface IsometricMapProps {
  tileConfig: TileConfig;
  viewState: ViewState;
  onViewStateChange: (params: { viewState: ViewState }) => void;
  lightDirection: [number, number, number];
  onTileHover: (tile: { x: number; y: number } | null) => void;
  scanlines?: ScanlineSettings;
  waterShader?: WaterShaderSettings;
}

export function IsometricMap({
  tileConfig,
  viewState,
  onViewStateChange,
  onTileHover,
  scanlines = { enabled: true, count: 480, opacity: 0.15 },
  waterShader,
}: IsometricMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const pmtilesRef = useRef<PMTiles | null>(null);
  const isUpdatingFromOSD = useRef(false);
  const isUpdatingFromProps = useRef(false);

  const {
    gridWidth,
    gridHeight,
    tileSize,
    maxZoomLevel,
    pmtilesUrl,
    pmtilesZoomMap,
  } = tileConfig;

  // Total image dimensions in pixels
  const totalWidth = gridWidth * tileSize;
  const totalHeight = gridHeight * tileSize;

  // Convert our view state to OSD viewport coordinates
  // Our viewState: { target: [worldX, worldY, 0], zoom: log2Scale }
  // OSD viewport: center is in image coordinates (0-1 for x, 0-aspectRatio for y)
  const worldToOsd = useCallback(
    (vs: ViewState) => {
      // Our target is in world pixels, convert to normalized coordinates
      // Note: Our Y=0 is at bottom, OSD Y=0 is at top
      const centerX = vs.target[0] / totalWidth;
      const centerY = 1 - vs.target[1] / totalHeight;

      // Our zoom: 0 = 1:1 pixels, positive = zoom in, negative = zoom out
      // OSD zoom: 1 = fit width in viewport
      // At zoom=0, we want 1 pixel = 1 pixel on screen
      // So OSD zoom = viewport_width / total_width * 2^ourZoom
      const scale = Math.pow(2, vs.zoom);
      const osdZoom = (window.innerWidth / totalWidth) * scale;

      return { centerX, centerY, zoom: osdZoom };
    },
    [totalWidth, totalHeight]
  );

  // Convert OSD viewport to our view state
  const osdToWorld = useCallback(
    (viewer: OpenSeadragon.Viewer): ViewState => {
      const viewport = viewer.viewport;
      const center = viewport.getCenter();
      const osdZoom = viewport.getZoom();

      // Convert normalized coordinates back to world pixels
      // OSD Y is top-down, ours is bottom-up
      const worldX = center.x * totalWidth;
      const worldY = (1 - center.y) * totalHeight;

      // Convert OSD zoom to our zoom
      // osdZoom = (windowWidth / totalWidth) * 2^ourZoom
      // ourZoom = log2(osdZoom * totalWidth / windowWidth)
      const ourZoom = Math.log2((osdZoom * totalWidth) / window.innerWidth);

      return {
        target: [worldX, worldY, 0],
        zoom: ourZoom,
      };
    },
    [totalWidth, totalHeight]
  );

  // Initialize PMTiles source if URL is provided
  useEffect(() => {
    if (pmtilesUrl && !pmtilesRef.current) {
      pmtilesRef.current = new PMTiles(pmtilesUrl);
    }
    return () => {
      pmtilesRef.current = null;
    };
  }, [pmtilesUrl]);

  // Initialize OpenSeadragon
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    // Calculate initial OSD viewport from our view state
    const { centerX, centerY, zoom: initialZoom } = worldToOsd(viewState);

    // Create tile source configuration
    // OSD pyramid: level 0 = least detail (few tiles), maxLevel = most detail (many tiles)
    // Our export: level 0 = most detail (128×128), level 4 = least detail (8×8)
    // PMTiles z: matches our level (0 = most detail when stored, but we flip it)
    // So we invert: ourLevel = maxZoomLevel - osdLevel
    const tileSourceConfig: OpenSeadragon.TileSourceOptions = {
      width: totalWidth,
      height: totalHeight,
      tileSize: tileSize,
      tileOverlap: 0,
      minLevel: 0,
      maxLevel: maxZoomLevel,
    };

    // If using PMTiles, we need a custom tile loading approach
    if (pmtilesUrl) {
      // For PMTiles, we use getTileUrl to generate a virtual URL
      // and downloadTileStart/downloadTileAbort to handle actual loading
      Object.assign(tileSourceConfig, {
        getTileUrl: (level: number, x: number, y: number) => {
          // Return a virtual URL that encodes the tile coordinates
          // This will be parsed by our custom tile loading
          const ourLevel = maxZoomLevel - level;

          // Get the PMTiles zoom level from the map, or calculate it
          let pmtilesZ: number;
          if (pmtilesZoomMap && pmtilesZoomMap[ourLevel] !== undefined) {
            pmtilesZ = pmtilesZoomMap[ourLevel];
          } else {
            // Fallback: calculate based on grid size
            const scale = Math.pow(2, ourLevel);
            const maxDim = Math.max(
              Math.ceil(gridWidth / scale),
              Math.ceil(gridHeight / scale)
            );
            pmtilesZ = maxDim <= 1 ? 0 : Math.ceil(Math.log2(maxDim));
          }

          return `pmtiles://${pmtilesZ}/${x}/${y}`;
        },
      });
    } else {
      // Legacy file-based tiles
      Object.assign(tileSourceConfig, {
        getTileUrl: (level: number, x: number, y: number) => {
          // Invert level mapping: OSD level 0 -> our level maxZoomLevel
          const ourLevel = maxZoomLevel - level;

          // Calculate grid dimensions at this level
          const scale = Math.pow(2, ourLevel);
          const levelGridWidth = Math.ceil(gridWidth / scale);
          const levelGridHeight = Math.ceil(gridHeight / scale);

          // Bounds check for this level
          if (x < 0 || x >= levelGridWidth || y < 0 || y >= levelGridHeight) {
            return "";
          }

          return `${import.meta.env.BASE_URL}tiles/${ourLevel}/${x}_${y}.png`;
        },
      });
    }

    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: "",
      showNavigationControl: false,
      showNavigator: false,
      animationTime: 0.15,
      blendTime: 0.1,
      minZoomImageRatio: 0.1,
      maxZoomPixelRatio: 16,
      visibilityRatio: 0.2,
      constrainDuringPan: false,
      gestureSettingsMouse: {
        scrollToZoom: true,
        clickToZoom: false,
        dblClickToZoom: true,
        flickEnabled: true,
      },
      gestureSettingsTouch: {
        scrollToZoom: false,
        clickToZoom: false,
        dblClickToZoom: true,
        flickEnabled: true,
        pinchToZoom: true,
      },
      // Disable image smoothing for pixel art
      imageSmoothingEnabled: false,
      tileSources: tileSourceConfig,
    });

    // If using PMTiles, set up custom tile downloading
    if (pmtilesUrl && pmtilesRef.current) {
      const pmtiles = pmtilesRef.current;

      // Override the tile downloading for PMTiles
      viewer.addHandler("open", () => {
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          tiledImage.setCompositeOperation("source-over");

          // Get the tile source and override its download method
          const source = tiledImage.source as OpenSeadragon.TileSource;

          // Store original methods
          const originalDownloadTileStart = (
            source as unknown as Record<string, unknown>
          ).downloadTileStart;

          // Override downloadTileStart to use PMTiles
          (source as unknown as Record<string, unknown>).downloadTileStart = (
            context: {
              src: string;
              finish: (
                data: HTMLImageElement | null,
                request: null,
                errorMsg?: string
              ) => void;
            }
          ) => {
            const url = context.src;

            // Check if this is a PMTiles URL
            if (url.startsWith("pmtiles://")) {
              const parts = url.replace("pmtiles://", "").split("/");
              const z = parseInt(parts[0], 10);
              const x = parseInt(parts[1], 10);
              const y = parseInt(parts[2], 10);

              // Fetch tile from PMTiles
              pmtiles
                .getZxy(z, x, y)
                .then((response) => {
                  if (response && response.data) {
                    // Convert ArrayBuffer to Blob URL
                    const blob = new Blob([response.data], {
                      type: "image/png",
                    });
                    const blobUrl = URL.createObjectURL(blob);

                    // Create an image element
                    const img = new Image();
                    img.onload = () => {
                      URL.revokeObjectURL(blobUrl);
                      context.finish(img, null);
                    };
                    img.onerror = () => {
                      URL.revokeObjectURL(blobUrl);
                      context.finish(null, null, "Failed to load tile image");
                    };
                    img.src = blobUrl;
                  } else {
                    // Tile not found, return null (transparent)
                    context.finish(null, null);
                  }
                })
                .catch((err) => {
                  console.error("PMTiles fetch error:", err);
                  context.finish(null, null, String(err));
                });
            } else if (typeof originalDownloadTileStart === "function") {
              // Fall back to original method for non-PMTiles URLs
              originalDownloadTileStart.call(source, context);
            }
          };
        }
      });
    }

    // Set initial viewport position (only for non-PMTiles mode,
    // PMTiles mode sets this in its own "open" handler above)
    if (!pmtilesUrl) {
      viewer.addHandler("open", () => {
        // Disable interpolation for crisp pixels
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          tiledImage.setCompositeOperation("source-over");
        }

        // Set initial position
        viewer.viewport.zoomTo(initialZoom, undefined, true);
        viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), true);
      });
    } else {
      // For PMTiles, add another handler for initial position
      // (the tile downloading setup handler already runs first)
      viewer.addHandler("open", () => {
        viewer.viewport.zoomTo(initialZoom, undefined, true);
        viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), true);
      });
    }

    // Track viewport changes
    viewer.addHandler("viewport-change", () => {
      if (isUpdatingFromProps.current) return;

      isUpdatingFromOSD.current = true;
      const newViewState = osdToWorld(viewer);
      onViewStateChange({ viewState: newViewState });
      isUpdatingFromOSD.current = false;
    });

    // Track mouse position for tile hover
    viewer.addHandler("canvas-exit", () => {
      onTileHover(null);
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMouseMove = (event: any) => {
      if (!event.position) return;
      const pos = event.position as OpenSeadragon.Point;

      const viewportPoint = viewer.viewport.pointFromPixel(pos);
      const imagePoint =
        viewer.viewport.viewportToImageCoordinates(viewportPoint);

      const tileX = Math.floor(imagePoint.x / tileSize);
      const tileY = Math.floor(imagePoint.y / tileSize);

      if (tileX >= 0 && tileX < gridWidth && tileY >= 0 && tileY < gridHeight) {
        onTileHover({ x: tileX, y: tileY });
      } else {
        onTileHover(null);
      }
    };

    viewer.addHandler("canvas-drag", handleMouseMove);
    viewer.addHandler("canvas-scroll", handleMouseMove);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (viewer as any).innerTracker.moveHandler = (event: any) => {
      handleMouseMove(event);
    };

    viewerRef.current = viewer;

    return () => {
      viewer.destroy();
      viewerRef.current = null;
    };
  }, [
    gridWidth,
    gridHeight,
    tileSize,
    maxZoomLevel,
    totalWidth,
    totalHeight,
    worldToOsd,
    osdToWorld,
    onViewStateChange,
    onTileHover,
    pmtilesUrl,
    pmtilesZoomMap,
  ]);

  // Sync external view state changes to OSD
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !viewer.viewport || isUpdatingFromOSD.current) return;

    isUpdatingFromProps.current = true;

    const { centerX, centerY, zoom } = worldToOsd(viewState);

    viewer.viewport.zoomTo(zoom, undefined, false);
    viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), false);

    isUpdatingFromProps.current = false;
  }, [viewState, worldToOsd]);

  return (
    <div className="map-container">
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          background: "#0a0c14",
        }}
      />
      {waterShader && (
        <WaterShaderOverlay
          enabled={waterShader.enabled}
          viewState={viewState}
          tileConfig={tileConfig}
          shaderParams={waterShader.params}
          showMask={waterShader.showMask}
        />
      )}
      <ScanlineOverlay
        enabled={scanlines.enabled}
        scanlineCount={scanlines.count}
        scanlineOpacity={scanlines.opacity}
      />
    </div>
  );
}

```

```typescriptreact
import { useEffect, useRef, useState } from "react";
import type { ViewState } from "../App";
import {
  initWebGL,
  createTexture,
  createSolidTexture,
  type ShaderParams,
  type ShaderLocations,
} from "../shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  tileSize: number;
  maxZoomLevel: number;
}

interface WaterShaderOverlayProps {
  enabled: boolean;
  viewState: ViewState;
  tileConfig: TileConfig;
  shaderParams: ShaderParams;
  showMask?: boolean;
}

// Cache for loaded textures (module-level to persist across re-renders)
const imageCache = new Map<string, WebGLTexture>();
const maskCache = new Map<string, WebGLTexture>();
const pendingLoads = new Set<string>();
const failedLoads = new Set<string>();

export function WaterShaderOverlay({
  enabled,
  viewState,
  tileConfig,
  shaderParams,
  showMask = false,
}: WaterShaderOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glContextRef = useRef<{
    gl: WebGLRenderingContext;
    program: WebGLProgram;
    locations: ShaderLocations;
  } | null>(null);
  const animationFrameRef = useRef<number>(0);
  const startTimeRef = useRef<number>(performance.now());
  const landTextureRef = useRef<WebGLTexture | null>(null);
  const buffersRef = useRef<{
    position: WebGLBuffer | null;
    texCoord: WebGLBuffer | null;
  }>({ position: null, texCoord: null });

  // Force re-render when textures load
  const [textureLoadCount, setTextureLoadCount] = useState(0);

  // Initialize WebGL context
  useEffect(() => {
    if (!enabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const context = initWebGL(canvas);
    if (!context) {
      console.error("WaterShaderOverlay: Failed to initialize WebGL");
      return;
    }

    console.log("WaterShaderOverlay: WebGL initialized successfully");
    glContextRef.current = context;
    const { gl, program } = context;

    // Create position buffer for a quad
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    buffersRef.current.position = positionBuffer;

    // Create texcoord buffer
    const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]);
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    buffersRef.current.texCoord = texCoordBuffer;

    // Create land (black) texture for tiles without masks
    landTextureRef.current = createSolidTexture(gl, [0, 0, 0, 255]);

    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const handleResize = () => {
      if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled]);

  // Load texture helper
  const loadTexture = (url: string, isImage: boolean) => {
    const cache = isImage ? imageCache : maskCache;

    if (cache.has(url) || pendingLoads.has(url) || failedLoads.has(url)) {
      return;
    }

    pendingLoads.add(url);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      pendingLoads.delete(url);
      const context = glContextRef.current;
      if (context) {
        const texture = createTexture(context.gl, img);
        if (texture) {
          cache.set(url, texture);
          setTextureLoadCount((c) => c + 1);
        }
      }
    };
    img.onerror = () => {
      pendingLoads.delete(url);
      failedLoads.add(url);
    };
    img.src = url;
  };

  // Animation loop
  useEffect(() => {
    if (!enabled) return;

    const render = () => {
      const context = glContextRef.current;
      const canvas = canvasRef.current;

      if (!context || !canvas) {
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }

      const { gl, program, locations } = context;
      const { gridWidth, gridHeight, tileSize } = tileConfig;
      const { target, zoom } = viewState;
      const elapsed = (performance.now() - startTimeRef.current) / 1000;

      // Update canvas size if needed
      if (
        canvas.width !== window.innerWidth ||
        canvas.height !== window.innerHeight
      ) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }

      const scale = Math.pow(2, zoom);
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const totalHeight = gridHeight * tileSize;

      // Water shader always uses level 0 (base tiles) since masks only exist at that level
      // At very zoomed out views, we skip water effects entirely for performance
      const skipWaterEffects = zoom < -3; // Skip when very zoomed out

      if (skipWaterEffects) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }

      // Convert viewState from Y-up (bottom=0) to Y-down (top=0) for OSD compatibility
      // Our viewState.target[1] is in Y-up coordinates
      const viewCenterX = target[0];
      const viewCenterY_down = totalHeight - target[1]; // Convert to Y-down

      // Calculate visible world bounds in Y-down coordinates
      const worldLeft = viewCenterX - viewportWidth / (2 * scale);
      const worldRight = viewCenterX + viewportWidth / (2 * scale);
      const worldTop = viewCenterY_down - viewportHeight / (2 * scale);
      const worldBottom = viewCenterY_down + viewportHeight / (2 * scale);

      // Calculate visible tile range (always using base level 0 tiles)
      const startX = Math.max(0, Math.floor(worldLeft / tileSize));
      const endX = Math.min(gridWidth - 1, Math.ceil(worldRight / tileSize));
      const startY = Math.max(0, Math.floor(worldTop / tileSize));
      const endY = Math.min(gridHeight - 1, Math.ceil(worldBottom / tileSize));

      // Clear the canvas
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);

      // Render each visible tile (always using level 0)
      for (let tileX = startX; tileX <= endX; tileX++) {
        for (let tileY = startY; tileY <= endY; tileY++) {
          // Always use level 0 tiles for both images and masks
          const imageUrl = `${import.meta.env.BASE_URL}tiles/0/${tileX}_${tileY}.png`;
          const maskUrl = `/water_masks/0/${tileX}_${tileY}.png`;

          // Get textures from cache
          const imageTexture = imageCache.get(imageUrl);
          const maskTexture = maskCache.get(maskUrl) || landTextureRef.current;

          // Start loading if not cached
          if (!imageTexture) {
            loadTexture(imageUrl, true);
            continue;
          }

          if (!maskCache.has(maskUrl) && !failedLoads.has(maskUrl)) {
            loadTexture(maskUrl, false);
          }

          if (!imageTexture || !maskTexture) continue;

          // Calculate screen position for this tile
          const tileWorldX = tileX * tileSize;
          const tileWorldY = tileY * tileSize; // Y-down coordinates

          // Convert to screen coordinates
          // Screen X: (worldX - viewCenterX) * scale + viewportWidth / 2
          // Screen Y: (worldY - viewCenterY_down) * scale + viewportHeight / 2
          // Note: WebGL canvas Y=0 is at bottom, so we need to flip for gl.viewport
          const screenX =
            (tileWorldX - viewCenterX) * scale + viewportWidth / 2;
          const screenY_fromTop =
            (tileWorldY - viewCenterY_down) * scale + viewportHeight / 2;
          const screenSize = tileSize * scale;

          // Convert to WebGL coordinates (Y=0 at bottom)
          const screenY = viewportHeight - screenY_fromTop - screenSize;

          // Skip if completely off-screen
          if (
            screenX + screenSize < 0 ||
            screenX > viewportWidth ||
            screenY + screenSize < 0 ||
            screenY > viewportHeight
          ) {
            continue;
          }

          // Set viewport and scissor for this tile
          gl.viewport(screenX, screenY, screenSize, screenSize);
          gl.scissor(screenX, screenY, screenSize, screenSize);
          gl.enable(gl.SCISSOR_TEST);

          // Bind buffers
          gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.position);
          gl.enableVertexAttribArray(locations.a_position);
          gl.vertexAttribPointer(
            locations.a_position,
            2,
            gl.FLOAT,
            false,
            0,
            0
          );

          gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.texCoord);
          gl.enableVertexAttribArray(locations.a_texCoord);
          gl.vertexAttribPointer(
            locations.a_texCoord,
            2,
            gl.FLOAT,
            false,
            0,
            0
          );

          // Set uniforms
          gl.uniform1f(locations.u_time, elapsed);
          gl.uniform1f(locations.u_waveSpeed, shaderParams.waveSpeed);
          gl.uniform1f(locations.u_waveFrequency, shaderParams.waveFrequency);
          gl.uniform1f(locations.u_foamThreshold, shaderParams.foamThreshold);
          gl.uniform1f(locations.u_pixelSize, shaderParams.pixelSize);
          gl.uniform2f(locations.u_resolution, screenSize, screenSize);
          gl.uniform1i(locations.u_showMask, showMask ? 1 : 0);
          gl.uniform1f(locations.u_rippleDarkness, shaderParams.rippleDarkness);
          gl.uniform1f(locations.u_waterDarkness, shaderParams.waterDarkness);

          // Bind textures
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, imageTexture);
          gl.uniform1i(locations.u_image, 0);

          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, maskTexture);
          gl.uniform1i(locations.u_mask, 1);

          // Draw
          gl.drawArrays(gl.TRIANGLES, 0, 6);
          gl.disable(gl.SCISSOR_TEST);
        }
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    animationFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [
    enabled,
    viewState,
    tileConfig,
    shaderParams,
    showMask,
    textureLoadCount,
  ]);

  if (!enabled) return null;

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 10,
      }}
    />
  );
}

```

Done! Updated the following:

1. **`vite.config.ts`** - Set `base: "/isometric-nyc/"` for production builds
2. **`App.tsx`** - Updated PMTiles URL, manifest URL, and tile pattern to use `import.meta.env.BASE_URL`
3. **`IsometricMap.tsx`** - Updated legacy tile URL
4. **`WaterShaderOverlay.tsx`** - Updated water mask tile URL

Now when you build:
- **Dev** (`bun run dev`): Uses `/` as base (localhost:3000)
- **Prod** (`bun run build`): Uses `/isometric-nyc/` as base (cannoneyed.com/isometric-nyc)

---

**User**

wait - but the tiles are served via R2 so they need a different URL base, right?

---

**Cursor**

You're absolutely right! The tiles come from R2, not GitHub Pages. Let me fix this:

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => ({
  // Use /isometric-nyc/ base path for production (GitHub Pages)
  base: mode === "production" ? "/isometric-nyc/" : "/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  server: {
    port: 3000,
    open: true,
  },
  define: {
    // R2 public URL for tiles (empty string = same origin, for dev)
    __TILES_BASE_URL__: JSON.stringify(
      mode === "production"
        ? "https://pub-f0fa143327b64a1bac21454a9c053597.r2.dev"
        : ""
    ),
  },
}));

```

```typescript
/// <reference types="vite/client" />

// Defined in vite.config.ts
declare const __TILES_BASE_URL__: string;

```

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = `${__TILES_BASE_URL__}/tiles.pmtiles`;

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch(`${import.meta.env.BASE_URL}tiles/manifest.json`)
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `${import.meta.env.BASE_URL}tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {isDev && (
        <ControlPanel
          scanlines={scanlines}
          onScanlinesChange={setScanlines}
          waterShader={waterShader}
          onWaterShaderChange={setWaterShader}
        />
      )}

      {isDev && <TileInfo hoveredTile={hoveredTile} viewState={viewState} />}
    </div>
  );
}

export default App;

```

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = `${__TILES_BASE_URL__}/tiles.pmtiles`;

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch(`${__TILES_BASE_URL__}/tiles/manifest.json`)
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `${import.meta.env.BASE_URL}tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {isDev && (
        <ControlPanel
          scanlines={scanlines}
          onScanlinesChange={setScanlines}
          waterShader={waterShader}
          onWaterShaderChange={setWaterShader}
        />
      )}

      {isDev && <TileInfo hoveredTile={hoveredTile} viewState={viewState} />}
    </div>
  );
}

export default App;

```

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = `${__TILES_BASE_URL__}/tiles.pmtiles`;

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch(`${__TILES_BASE_URL__}/tiles/manifest.json`)
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `${__TILES_BASE_URL__}/tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {isDev && (
        <ControlPanel
          scanlines={scanlines}
          onScanlinesChange={setScanlines}
          waterShader={waterShader}
          onWaterShaderChange={setWaterShader}
        />
      )}

      {isDev && <TileInfo hoveredTile={hoveredTile} viewState={viewState} />}
    </div>
  );
}

export default App;

```

```typescriptreact
import { useEffect, useRef, useCallback } from "react";
import OpenSeadragon from "openseadragon";
import { PMTiles } from "pmtiles";
import type { ViewState } from "../App";
import { ScanlineOverlay } from "./ScanlineOverlay";
import { WaterShaderOverlay } from "./WaterShaderOverlay";
import type { ShaderParams } from "../shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file (optional, falls back to tile directory)
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

interface ScanlineSettings {
  enabled: boolean;
  count: number;
  opacity: number;
}

interface WaterShaderSettings {
  enabled: boolean;
  showMask: boolean;
  params: ShaderParams;
}

interface IsometricMapProps {
  tileConfig: TileConfig;
  viewState: ViewState;
  onViewStateChange: (params: { viewState: ViewState }) => void;
  lightDirection: [number, number, number];
  onTileHover: (tile: { x: number; y: number } | null) => void;
  scanlines?: ScanlineSettings;
  waterShader?: WaterShaderSettings;
}

export function IsometricMap({
  tileConfig,
  viewState,
  onViewStateChange,
  onTileHover,
  scanlines = { enabled: true, count: 480, opacity: 0.15 },
  waterShader,
}: IsometricMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const pmtilesRef = useRef<PMTiles | null>(null);
  const isUpdatingFromOSD = useRef(false);
  const isUpdatingFromProps = useRef(false);

  const {
    gridWidth,
    gridHeight,
    tileSize,
    maxZoomLevel,
    pmtilesUrl,
    pmtilesZoomMap,
  } = tileConfig;

  // Total image dimensions in pixels
  const totalWidth = gridWidth * tileSize;
  const totalHeight = gridHeight * tileSize;

  // Convert our view state to OSD viewport coordinates
  // Our viewState: { target: [worldX, worldY, 0], zoom: log2Scale }
  // OSD viewport: center is in image coordinates (0-1 for x, 0-aspectRatio for y)
  const worldToOsd = useCallback(
    (vs: ViewState) => {
      // Our target is in world pixels, convert to normalized coordinates
      // Note: Our Y=0 is at bottom, OSD Y=0 is at top
      const centerX = vs.target[0] / totalWidth;
      const centerY = 1 - vs.target[1] / totalHeight;

      // Our zoom: 0 = 1:1 pixels, positive = zoom in, negative = zoom out
      // OSD zoom: 1 = fit width in viewport
      // At zoom=0, we want 1 pixel = 1 pixel on screen
      // So OSD zoom = viewport_width / total_width * 2^ourZoom
      const scale = Math.pow(2, vs.zoom);
      const osdZoom = (window.innerWidth / totalWidth) * scale;

      return { centerX, centerY, zoom: osdZoom };
    },
    [totalWidth, totalHeight]
  );

  // Convert OSD viewport to our view state
  const osdToWorld = useCallback(
    (viewer: OpenSeadragon.Viewer): ViewState => {
      const viewport = viewer.viewport;
      const center = viewport.getCenter();
      const osdZoom = viewport.getZoom();

      // Convert normalized coordinates back to world pixels
      // OSD Y is top-down, ours is bottom-up
      const worldX = center.x * totalWidth;
      const worldY = (1 - center.y) * totalHeight;

      // Convert OSD zoom to our zoom
      // osdZoom = (windowWidth / totalWidth) * 2^ourZoom
      // ourZoom = log2(osdZoom * totalWidth / windowWidth)
      const ourZoom = Math.log2((osdZoom * totalWidth) / window.innerWidth);

      return {
        target: [worldX, worldY, 0],
        zoom: ourZoom,
      };
    },
    [totalWidth, totalHeight]
  );

  // Initialize PMTiles source if URL is provided
  useEffect(() => {
    if (pmtilesUrl && !pmtilesRef.current) {
      pmtilesRef.current = new PMTiles(pmtilesUrl);
    }
    return () => {
      pmtilesRef.current = null;
    };
  }, [pmtilesUrl]);

  // Initialize OpenSeadragon
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    // Calculate initial OSD viewport from our view state
    const { centerX, centerY, zoom: initialZoom } = worldToOsd(viewState);

    // Create tile source configuration
    // OSD pyramid: level 0 = least detail (few tiles), maxLevel = most detail (many tiles)
    // Our export: level 0 = most detail (128×128), level 4 = least detail (8×8)
    // PMTiles z: matches our level (0 = most detail when stored, but we flip it)
    // So we invert: ourLevel = maxZoomLevel - osdLevel
    const tileSourceConfig: OpenSeadragon.TileSourceOptions = {
      width: totalWidth,
      height: totalHeight,
      tileSize: tileSize,
      tileOverlap: 0,
      minLevel: 0,
      maxLevel: maxZoomLevel,
    };

    // If using PMTiles, we need a custom tile loading approach
    if (pmtilesUrl) {
      // For PMTiles, we use getTileUrl to generate a virtual URL
      // and downloadTileStart/downloadTileAbort to handle actual loading
      Object.assign(tileSourceConfig, {
        getTileUrl: (level: number, x: number, y: number) => {
          // Return a virtual URL that encodes the tile coordinates
          // This will be parsed by our custom tile loading
          const ourLevel = maxZoomLevel - level;

          // Get the PMTiles zoom level from the map, or calculate it
          let pmtilesZ: number;
          if (pmtilesZoomMap && pmtilesZoomMap[ourLevel] !== undefined) {
            pmtilesZ = pmtilesZoomMap[ourLevel];
          } else {
            // Fallback: calculate based on grid size
            const scale = Math.pow(2, ourLevel);
            const maxDim = Math.max(
              Math.ceil(gridWidth / scale),
              Math.ceil(gridHeight / scale)
            );
            pmtilesZ = maxDim <= 1 ? 0 : Math.ceil(Math.log2(maxDim));
          }

          return `pmtiles://${pmtilesZ}/${x}/${y}`;
        },
      });
    } else {
      // Legacy file-based tiles
      Object.assign(tileSourceConfig, {
        getTileUrl: (level: number, x: number, y: number) => {
          // Invert level mapping: OSD level 0 -> our level maxZoomLevel
          const ourLevel = maxZoomLevel - level;

          // Calculate grid dimensions at this level
          const scale = Math.pow(2, ourLevel);
          const levelGridWidth = Math.ceil(gridWidth / scale);
          const levelGridHeight = Math.ceil(gridHeight / scale);

          // Bounds check for this level
          if (x < 0 || x >= levelGridWidth || y < 0 || y >= levelGridHeight) {
            return "";
          }

          return `${__TILES_BASE_URL__}/tiles/${ourLevel}/${x}_${y}.png`;
        },
      });
    }

    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: "",
      showNavigationControl: false,
      showNavigator: false,
      animationTime: 0.15,
      blendTime: 0.1,
      minZoomImageRatio: 0.1,
      maxZoomPixelRatio: 16,
      visibilityRatio: 0.2,
      constrainDuringPan: false,
      gestureSettingsMouse: {
        scrollToZoom: true,
        clickToZoom: false,
        dblClickToZoom: true,
        flickEnabled: true,
      },
      gestureSettingsTouch: {
        scrollToZoom: false,
        clickToZoom: false,
        dblClickToZoom: true,
        flickEnabled: true,
        pinchToZoom: true,
      },
      // Disable image smoothing for pixel art
      imageSmoothingEnabled: false,
      tileSources: tileSourceConfig,
    });

    // If using PMTiles, set up custom tile downloading
    if (pmtilesUrl && pmtilesRef.current) {
      const pmtiles = pmtilesRef.current;

      // Override the tile downloading for PMTiles
      viewer.addHandler("open", () => {
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          tiledImage.setCompositeOperation("source-over");

          // Get the tile source and override its download method
          const source = tiledImage.source as OpenSeadragon.TileSource;

          // Store original methods
          const originalDownloadTileStart = (
            source as unknown as Record<string, unknown>
          ).downloadTileStart;

          // Override downloadTileStart to use PMTiles
          (source as unknown as Record<string, unknown>).downloadTileStart = (
            context: {
              src: string;
              finish: (
                data: HTMLImageElement | null,
                request: null,
                errorMsg?: string
              ) => void;
            }
          ) => {
            const url = context.src;

            // Check if this is a PMTiles URL
            if (url.startsWith("pmtiles://")) {
              const parts = url.replace("pmtiles://", "").split("/");
              const z = parseInt(parts[0], 10);
              const x = parseInt(parts[1], 10);
              const y = parseInt(parts[2], 10);

              // Fetch tile from PMTiles
              pmtiles
                .getZxy(z, x, y)
                .then((response) => {
                  if (response && response.data) {
                    // Convert ArrayBuffer to Blob URL
                    const blob = new Blob([response.data], {
                      type: "image/png",
                    });
                    const blobUrl = URL.createObjectURL(blob);

                    // Create an image element
                    const img = new Image();
                    img.onload = () => {
                      URL.revokeObjectURL(blobUrl);
                      context.finish(img, null);
                    };
                    img.onerror = () => {
                      URL.revokeObjectURL(blobUrl);
                      context.finish(null, null, "Failed to load tile image");
                    };
                    img.src = blobUrl;
                  } else {
                    // Tile not found, return null (transparent)
                    context.finish(null, null);
                  }
                })
                .catch((err) => {
                  console.error("PMTiles fetch error:", err);
                  context.finish(null, null, String(err));
                });
            } else if (typeof originalDownloadTileStart === "function") {
              // Fall back to original method for non-PMTiles URLs
              originalDownloadTileStart.call(source, context);
            }
          };
        }
      });
    }

    // Set initial viewport position (only for non-PMTiles mode,
    // PMTiles mode sets this in its own "open" handler above)
    if (!pmtilesUrl) {
      viewer.addHandler("open", () => {
        // Disable interpolation for crisp pixels
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          tiledImage.setCompositeOperation("source-over");
        }

        // Set initial position
        viewer.viewport.zoomTo(initialZoom, undefined, true);
        viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), true);
      });
    } else {
      // For PMTiles, add another handler for initial position
      // (the tile downloading setup handler already runs first)
      viewer.addHandler("open", () => {
        viewer.viewport.zoomTo(initialZoom, undefined, true);
        viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), true);
      });
    }

    // Track viewport changes
    viewer.addHandler("viewport-change", () => {
      if (isUpdatingFromProps.current) return;

      isUpdatingFromOSD.current = true;
      const newViewState = osdToWorld(viewer);
      onViewStateChange({ viewState: newViewState });
      isUpdatingFromOSD.current = false;
    });

    // Track mouse position for tile hover
    viewer.addHandler("canvas-exit", () => {
      onTileHover(null);
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMouseMove = (event: any) => {
      if (!event.position) return;
      const pos = event.position as OpenSeadragon.Point;

      const viewportPoint = viewer.viewport.pointFromPixel(pos);
      const imagePoint =
        viewer.viewport.viewportToImageCoordinates(viewportPoint);

      const tileX = Math.floor(imagePoint.x / tileSize);
      const tileY = Math.floor(imagePoint.y / tileSize);

      if (tileX >= 0 && tileX < gridWidth && tileY >= 0 && tileY < gridHeight) {
        onTileHover({ x: tileX, y: tileY });
      } else {
        onTileHover(null);
      }
    };

    viewer.addHandler("canvas-drag", handleMouseMove);
    viewer.addHandler("canvas-scroll", handleMouseMove);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (viewer as any).innerTracker.moveHandler = (event: any) => {
      handleMouseMove(event);
    };

    viewerRef.current = viewer;

    return () => {
      viewer.destroy();
      viewerRef.current = null;
    };
  }, [
    gridWidth,
    gridHeight,
    tileSize,
    maxZoomLevel,
    totalWidth,
    totalHeight,
    worldToOsd,
    osdToWorld,
    onViewStateChange,
    onTileHover,
    pmtilesUrl,
    pmtilesZoomMap,
  ]);

  // Sync external view state changes to OSD
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !viewer.viewport || isUpdatingFromOSD.current) return;

    isUpdatingFromProps.current = true;

    const { centerX, centerY, zoom } = worldToOsd(viewState);

    viewer.viewport.zoomTo(zoom, undefined, false);
    viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), false);

    isUpdatingFromProps.current = false;
  }, [viewState, worldToOsd]);

  return (
    <div className="map-container">
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          background: "#0a0c14",
        }}
      />
      {waterShader && (
        <WaterShaderOverlay
          enabled={waterShader.enabled}
          viewState={viewState}
          tileConfig={tileConfig}
          shaderParams={waterShader.params}
          showMask={waterShader.showMask}
        />
      )}
      <ScanlineOverlay
        enabled={scanlines.enabled}
        scanlineCount={scanlines.count}
        scanlineOpacity={scanlines.opacity}
      />
    </div>
  );
}

```

```typescriptreact
import { useEffect, useRef, useState } from "react";
import type { ViewState } from "../App";
import {
  initWebGL,
  createTexture,
  createSolidTexture,
  type ShaderParams,
  type ShaderLocations,
} from "../shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  tileSize: number;
  maxZoomLevel: number;
}

interface WaterShaderOverlayProps {
  enabled: boolean;
  viewState: ViewState;
  tileConfig: TileConfig;
  shaderParams: ShaderParams;
  showMask?: boolean;
}

// Cache for loaded textures (module-level to persist across re-renders)
const imageCache = new Map<string, WebGLTexture>();
const maskCache = new Map<string, WebGLTexture>();
const pendingLoads = new Set<string>();
const failedLoads = new Set<string>();

export function WaterShaderOverlay({
  enabled,
  viewState,
  tileConfig,
  shaderParams,
  showMask = false,
}: WaterShaderOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glContextRef = useRef<{
    gl: WebGLRenderingContext;
    program: WebGLProgram;
    locations: ShaderLocations;
  } | null>(null);
  const animationFrameRef = useRef<number>(0);
  const startTimeRef = useRef<number>(performance.now());
  const landTextureRef = useRef<WebGLTexture | null>(null);
  const buffersRef = useRef<{
    position: WebGLBuffer | null;
    texCoord: WebGLBuffer | null;
  }>({ position: null, texCoord: null });

  // Force re-render when textures load
  const [textureLoadCount, setTextureLoadCount] = useState(0);

  // Initialize WebGL context
  useEffect(() => {
    if (!enabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const context = initWebGL(canvas);
    if (!context) {
      console.error("WaterShaderOverlay: Failed to initialize WebGL");
      return;
    }

    console.log("WaterShaderOverlay: WebGL initialized successfully");
    glContextRef.current = context;
    const { gl, program } = context;

    // Create position buffer for a quad
    const positions = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    buffersRef.current.position = positionBuffer;

    // Create texcoord buffer
    const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]);
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    buffersRef.current.texCoord = texCoordBuffer;

    // Create land (black) texture for tiles without masks
    landTextureRef.current = createSolidTexture(gl, [0, 0, 0, 255]);

    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const handleResize = () => {
      if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [enabled]);

  // Load texture helper
  const loadTexture = (url: string, isImage: boolean) => {
    const cache = isImage ? imageCache : maskCache;

    if (cache.has(url) || pendingLoads.has(url) || failedLoads.has(url)) {
      return;
    }

    pendingLoads.add(url);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      pendingLoads.delete(url);
      const context = glContextRef.current;
      if (context) {
        const texture = createTexture(context.gl, img);
        if (texture) {
          cache.set(url, texture);
          setTextureLoadCount((c) => c + 1);
        }
      }
    };
    img.onerror = () => {
      pendingLoads.delete(url);
      failedLoads.add(url);
    };
    img.src = url;
  };

  // Animation loop
  useEffect(() => {
    if (!enabled) return;

    const render = () => {
      const context = glContextRef.current;
      const canvas = canvasRef.current;

      if (!context || !canvas) {
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }

      const { gl, program, locations } = context;
      const { gridWidth, gridHeight, tileSize } = tileConfig;
      const { target, zoom } = viewState;
      const elapsed = (performance.now() - startTimeRef.current) / 1000;

      // Update canvas size if needed
      if (
        canvas.width !== window.innerWidth ||
        canvas.height !== window.innerHeight
      ) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }

      const scale = Math.pow(2, zoom);
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const totalHeight = gridHeight * tileSize;

      // Water shader always uses level 0 (base tiles) since masks only exist at that level
      // At very zoomed out views, we skip water effects entirely for performance
      const skipWaterEffects = zoom < -3; // Skip when very zoomed out

      if (skipWaterEffects) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        animationFrameRef.current = requestAnimationFrame(render);
        return;
      }

      // Convert viewState from Y-up (bottom=0) to Y-down (top=0) for OSD compatibility
      // Our viewState.target[1] is in Y-up coordinates
      const viewCenterX = target[0];
      const viewCenterY_down = totalHeight - target[1]; // Convert to Y-down

      // Calculate visible world bounds in Y-down coordinates
      const worldLeft = viewCenterX - viewportWidth / (2 * scale);
      const worldRight = viewCenterX + viewportWidth / (2 * scale);
      const worldTop = viewCenterY_down - viewportHeight / (2 * scale);
      const worldBottom = viewCenterY_down + viewportHeight / (2 * scale);

      // Calculate visible tile range (always using base level 0 tiles)
      const startX = Math.max(0, Math.floor(worldLeft / tileSize));
      const endX = Math.min(gridWidth - 1, Math.ceil(worldRight / tileSize));
      const startY = Math.max(0, Math.floor(worldTop / tileSize));
      const endY = Math.min(gridHeight - 1, Math.ceil(worldBottom / tileSize));

      // Clear the canvas
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);

      // Render each visible tile (always using level 0)
      for (let tileX = startX; tileX <= endX; tileX++) {
        for (let tileY = startY; tileY <= endY; tileY++) {
          // Always use level 0 tiles for both images and masks
          const imageUrl = `${__TILES_BASE_URL__}/tiles/0/${tileX}_${tileY}.png`;
          const maskUrl = `/water_masks/0/${tileX}_${tileY}.png`;

          // Get textures from cache
          const imageTexture = imageCache.get(imageUrl);
          const maskTexture = maskCache.get(maskUrl) || landTextureRef.current;

          // Start loading if not cached
          if (!imageTexture) {
            loadTexture(imageUrl, true);
            continue;
          }

          if (!maskCache.has(maskUrl) && !failedLoads.has(maskUrl)) {
            loadTexture(maskUrl, false);
          }

          if (!imageTexture || !maskTexture) continue;

          // Calculate screen position for this tile
          const tileWorldX = tileX * tileSize;
          const tileWorldY = tileY * tileSize; // Y-down coordinates

          // Convert to screen coordinates
          // Screen X: (worldX - viewCenterX) * scale + viewportWidth / 2
          // Screen Y: (worldY - viewCenterY_down) * scale + viewportHeight / 2
          // Note: WebGL canvas Y=0 is at bottom, so we need to flip for gl.viewport
          const screenX =
            (tileWorldX - viewCenterX) * scale + viewportWidth / 2;
          const screenY_fromTop =
            (tileWorldY - viewCenterY_down) * scale + viewportHeight / 2;
          const screenSize = tileSize * scale;

          // Convert to WebGL coordinates (Y=0 at bottom)
          const screenY = viewportHeight - screenY_fromTop - screenSize;

          // Skip if completely off-screen
          if (
            screenX + screenSize < 0 ||
            screenX > viewportWidth ||
            screenY + screenSize < 0 ||
            screenY > viewportHeight
          ) {
            continue;
          }

          // Set viewport and scissor for this tile
          gl.viewport(screenX, screenY, screenSize, screenSize);
          gl.scissor(screenX, screenY, screenSize, screenSize);
          gl.enable(gl.SCISSOR_TEST);

          // Bind buffers
          gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.position);
          gl.enableVertexAttribArray(locations.a_position);
          gl.vertexAttribPointer(
            locations.a_position,
            2,
            gl.FLOAT,
            false,
            0,
            0
          );

          gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.texCoord);
          gl.enableVertexAttribArray(locations.a_texCoord);
          gl.vertexAttribPointer(
            locations.a_texCoord,
            2,
            gl.FLOAT,
            false,
            0,
            0
          );

          // Set uniforms
          gl.uniform1f(locations.u_time, elapsed);
          gl.uniform1f(locations.u_waveSpeed, shaderParams.waveSpeed);
          gl.uniform1f(locations.u_waveFrequency, shaderParams.waveFrequency);
          gl.uniform1f(locations.u_foamThreshold, shaderParams.foamThreshold);
          gl.uniform1f(locations.u_pixelSize, shaderParams.pixelSize);
          gl.uniform2f(locations.u_resolution, screenSize, screenSize);
          gl.uniform1i(locations.u_showMask, showMask ? 1 : 0);
          gl.uniform1f(locations.u_rippleDarkness, shaderParams.rippleDarkness);
          gl.uniform1f(locations.u_waterDarkness, shaderParams.waterDarkness);

          // Bind textures
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, imageTexture);
          gl.uniform1i(locations.u_image, 0);

          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, maskTexture);
          gl.uniform1i(locations.u_mask, 1);

          // Draw
          gl.drawArrays(gl.TRIANGLES, 0, 6);
          gl.disable(gl.SCISSOR_TEST);
        }
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    animationFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [
    enabled,
    viewState,
    tileConfig,
    shaderParams,
    showMask,
    textureLoadCount,
  ]);

  if (!enabled) return null;

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 10,
      }}
    />
  );
}

```

Fixed! Now:
- **App HTML/JS/CSS**: Served from GitHub Pages at `/isometric-nyc/`
- **Tiles (PMTiles)**: Served from R2

**Important**: Update the R2 URL in `vite.config.ts` with your actual R2 public bucket URL:

```typescript
__TILES_BASE_URL__: JSON.stringify(
  mode === "production"
    ? "https://YOUR-R2-PUBLIC-URL.r2.dev"  // ← Replace this
    : ""
),
```

To get your R2 public URL:
1. Go to Cloudflare Dashboard → R2
2. Click on your `isometric-nyc` bucket
3. Go to **Settings** → **Public access**
4. Enable public access and copy the public URL

```typescript
__TILES_BASE_URL__: JSON.stringify(
  mode === "production"
    ? "https://YOUR-R2-PUBLIC-URL.r2.dev"  // ← Replace this
    : ""
),
```

---

**User**

There's no "public access", just a "public development url" and when I select it it says "This URL is rate-limited and not recommended for production. Cloudflare features like Access and Caching are unavailable. Connect a custom domain to the bucket to support production workloads."

---

**Cursor**

That's fine for now! The dev URL works for testing. For production, you have two options:

### Option 1: Custom Domain (Recommended)
Connect a subdomain to your R2 bucket:
1. In R2 bucket settings → **Custom domains**
2. Add something like `tiles.cannoneyed.com`
3. Cloudflare will set up the DNS automatically

Then update `vite.config.ts`:
```typescript
? "https://tiles.cannoneyed.com"
```

### Option 2: Use Dev URL for Now
The rate-limited URL you have is fine for testing and low-traffic use. You can always add a custom domain later.

---

For your initial deployment test, the dev URL (`pub-4d4013d62ff44fc6b63d825226ec07bd.r2.dev`) you already configured should work. Just keep in mind it has rate limits if you expect significant traffic.

```typescript
? "https://tiles.cannoneyed.com"
```

---

**User**

"Access to fetch at 'https://pub-4d4013d62ff44fc6b63d825226ec07bd.r2.dev/tiles.pmtiles' from origin 'https://cannoneyed.com' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource. If an opaque response serves your needs, set the request's mode to 'no-cors' to fetch the resource with CORS disabled."

---

**Cursor**

You need to configure CORS on your R2 bucket. Here's how:

1. Go to **Cloudflare Dashboard** → **R2** → **isometric-nyc** bucket
2. Click **Settings** tab
3. Scroll to **CORS Policy** and click **Add CORS policy**
4. Add this configuration:

```json
[
  {
    "AllowedOrigins": [
      "https://cannoneyed.com",
      "http://localhost:3000"
    ],
    "AllowedMethods": [
      "GET",
      "HEAD"
    ],
    "AllowedHeaders": [
      "*"
    ],
    "MaxAgeSeconds": 86400
  }
]
```

This allows:
- Your production site (`cannoneyed.com`)
- Local development (`localhost:3000`)
- GET and HEAD requests (all you need for tiles)
- Caches the CORS preflight for 24 hours

```json
[
  {
    "AllowedOrigins": [
      "https://cannoneyed.com",
      "http://localhost:3000"
    ],
    "AllowedMethods": [
      "GET",
      "HEAD"
    ],
    "AllowedHeaders": [
      "*"
    ],
    "MaxAgeSeconds": 86400
  }
]
```

---

**User**

It doesn't seem like the tiles.pmtiles has been uploaded!

---

**Cursor**

Let's upload it! Run:

```bash
uv run python src/app/deploy.py --r2
```

Or if you want to see what it would do first:

```bash
uv run python src/app/deploy.py --r2 --dry-run
```

Make sure:
1. You've run `wrangler login` to authenticate
2. The `tiles.pmtiles` file exists in `src/app/public/`

```shellscript
uv run python src/app/deploy.py --r2
```

```shellscript
uv run python src/app/deploy.py --r2 --dry-run
```

---

**User**

it says it worked but when I look at the dashboard it says 0 bytes

---

**Cursor**

Let me check the upload command directly:

Found it! Wrangler is uploading to a **local** bucket, not the remote R2. We need the `--remote` flag:

Now it uploaded to **remote**! Let me fix the deploy script to include `--remote`:

```python
#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing pmtiles to Cloudflare R2 (using wrangler)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.parent
DIST_DIR = APP_DIR / "dist"
PUBLIC_DIR = APP_DIR / "public"

# GitHub Pages configuration
GITHUB_PAGES_BRANCH = "gh-pages"
GITHUB_PAGES_REMOTE = "origin"

# Cloudflare R2 configuration
R2_BUCKET_NAME = "isometric-nyc"

# Files to skip during deployment
SKIP_FILES = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}


def should_skip_file(file_path: Path) -> bool:
  """Check if a file should be skipped during deployment."""
  return file_path.name in SKIP_FILES or file_path.name.startswith("._")


def run_command(
  cmd: list[str],
  cwd: Path | None = None,
  check: bool = True,
  capture_output: bool = False,
) -> subprocess.CompletedProcess:
  """Run a shell command and handle errors."""
  print(f"  → Running: {' '.join(cmd)}")
  result = subprocess.run(
    cmd,
    cwd=cwd,
    check=check,
    capture_output=capture_output,
    text=True,
  )
  return result


def build_app() -> bool:
  """Build the static web app using bun/vite."""
  print("\n📦 Building web app...")

  # Check if bun is available
  if shutil.which("bun") is None:
    print("  ❌ Error: bun is not installed or not in PATH")
    print("  Install bun: curl -fsSL https://bun.sh/install | bash")
    return False

  # Install dependencies if needed
  if not (APP_DIR / "node_modules").exists():
    print("  📥 Installing dependencies...")
    run_command(["bun", "install"], cwd=APP_DIR)

  # Run the build
  print("  🔨 Running build...")
  run_command(["bun", "run", "build"], cwd=APP_DIR)

  if DIST_DIR.exists():
    print(f"  ✅ Build complete: {DIST_DIR}")
    return True
  else:
    print("  ❌ Build failed: dist directory not created")
    return False


def deploy_github_pages(dry_run: bool = False) -> bool:
  """Deploy built content to GitHub Pages branch."""
  print("\n🌐 Deploying to GitHub Pages...")

  if not DIST_DIR.exists():
    print("  ❌ Error: dist directory does not exist. Run build first.")
    return False

  # Create a temporary directory for the gh-pages content
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Copy dist contents to temp directory (excluding large assets)
    print("  📋 Preparing deployment files...")
    for item in DIST_DIR.iterdir():
      # Skip tiles and pmtiles - they'll be served from R2
      if (
        item.name in ("tiles", "tiles_processed", "water_masks")
        or item.suffix == ".pmtiles"
      ):
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      # Skip junk files
      if should_skip_file(item):
        continue

      dest = tmp_path / item.name
      if item.is_dir():
        shutil.copytree(
          item,
          dest,
          ignore=shutil.ignore_patterns(*SKIP_FILES, "._*"),
        )
      else:
        shutil.copy2(item, dest)

    # Create .nojekyll file to disable Jekyll processing
    (tmp_path / ".nojekyll").touch()

    # Create CNAME file if a custom domain is configured
    custom_domain = os.environ.get("GITHUB_PAGES_DOMAIN")
    if custom_domain:
      (tmp_path / "CNAME").write_text(custom_domain)
      print(f"    📝 Created CNAME for {custom_domain}")

    if dry_run:
      print("  🔍 Dry run - would deploy these files:")
      for f in tmp_path.rglob("*"):
        if f.is_file():
          print(f"    - {f.relative_to(tmp_path)}")
      return True

    # Initialize git repo in temp directory
    print("  🔧 Initializing git repository...")
    run_command(["git", "init"], cwd=tmp_path)
    run_command(["git", "checkout", "-b", GITHUB_PAGES_BRANCH], cwd=tmp_path)

    # Get the remote URL from the main repo
    result = run_command(
      ["git", "remote", "get-url", GITHUB_PAGES_REMOTE],
      cwd=PROJECT_ROOT,
      capture_output=True,
    )
    remote_url = result.stdout.strip()

    run_command(["git", "remote", "add", GITHUB_PAGES_REMOTE, remote_url], cwd=tmp_path)

    # Add and commit all files
    run_command(["git", "add", "-A"], cwd=tmp_path)
    run_command(
      ["git", "commit", "-m", "Deploy to GitHub Pages"],
      cwd=tmp_path,
    )

    # Force push to gh-pages branch
    print("  🚀 Pushing to GitHub Pages...")
    run_command(
      ["git", "push", "--force", GITHUB_PAGES_REMOTE, GITHUB_PAGES_BRANCH],
      cwd=tmp_path,
    )

    print("  ✅ GitHub Pages deployment complete!")
    return True


def upload_to_r2(dry_run: bool = False) -> bool:
  """Upload pmtiles and tile assets to Cloudflare R2 using wrangler."""
  print("\n☁️  Uploading to Cloudflare R2...")

  # Check if wrangler is available
  if shutil.which("wrangler") is None:
    print("  ❌ Error: wrangler is not installed or not in PATH")
    print("  Install wrangler: bun add -g wrangler")
    print("  Then authenticate: wrangler login")
    return False

  # Find pmtiles file
  pmtiles_file = PUBLIC_DIR / "tiles.pmtiles"
  if not pmtiles_file.exists():
    pmtiles_file = DIST_DIR / "tiles.pmtiles"

  if not pmtiles_file.exists():
    print(f"  ⚠️  Warning: tiles.pmtiles not found in {PUBLIC_DIR} or {DIST_DIR}")
    pmtiles_file = None

  # Collect files to upload (only pmtiles)
  files_to_upload: list[tuple[Path, str]] = []

  if pmtiles_file:
    files_to_upload.append((pmtiles_file, "tiles.pmtiles"))

  if not files_to_upload:
    print("  ⚠️  No files found to upload")
    return True

  print(f"  📊 Found {len(files_to_upload)} files to upload")

  if dry_run:
    print("  🔍 Dry run - would upload these files:")
    for local_path, remote_key in files_to_upload[:20]:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    - {remote_key} ({size_mb:.2f} MB)")
    if len(files_to_upload) > 20:
      print(f"    ... and {len(files_to_upload) - 20} more files")
    return True

  # Upload files using wrangler
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      print(f"    ⬆️  Uploading {remote_key}...", end=" ", flush=True)

      # Use wrangler r2 object put (--remote to upload to actual R2, not local)
      result = subprocess.run(
        [
          "wrangler",
          "r2",
          "object",
          "put",
          f"{R2_BUCKET_NAME}/{remote_key}",
          "--file",
          str(local_path),
          "--remote",
        ],
        capture_output=True,
        text=True,
      )

      if result.returncode == 0:
        print("✓")
        uploaded += 1
      else:
        print(f"✗ ({result.stderr.strip()})")
        failed += 1

    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  print(f"\n  📊 Upload summary: {uploaded} succeeded, {failed} failed")

  if failed == 0:
    print("  ✅ R2 upload complete!")
    if pmtiles_file:
      print("  🔗 PMTiles will be available at your R2 public URL/tiles.pmtiles")
    return True
  else:
    print("  ⚠️  Some uploads failed")
    return False


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Prerequisites:
  - bun: Install with `curl -fsSL https://bun.sh/install | bash`
  - wrangler: Install with `bun add -g wrangler`, then run `wrangler login`

Environment Variables (optional):
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run

  # Just upload to R2
  python deploy.py --r2
    """,
  )

  parser.add_argument(
    "--build",
    action="store_true",
    help="Build the web app",
  )
  parser.add_argument(
    "--github-pages",
    action="store_true",
    help="Deploy to GitHub Pages",
  )
  parser.add_argument(
    "--r2",
    action="store_true",
    help="Upload to Cloudflare R2",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )

  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all:
    if not build_app():
      success = False
      if not args.all:
        return 1

  # GitHub Pages
  if args.github_pages or args.all:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  # R2 upload
  if args.r2 or args.all:
    if not upload_to_r2(dry_run=args.dry_run):
      success = False
      if not args.all:
        return 1

  print("\n" + "=" * 40)
  if success:
    print("✅ Deployment complete!")
  else:
    print("⚠️  Deployment completed with some errors")

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

Fixed! The issue was wrangler defaulting to local mode. Now the deploy script includes `--remote`.

Check your R2 dashboard now - the `tiles.pmtiles` (64 MB) should be there. Then try refreshing your site!

---

**User**

OK now I'm getting this error in the deployed app (and countless more) 11_12.png:1 
        
        
       GET https://pub-4d4013d62ff44fc6b63d825226ec07bd.r2.dev/tiles/0/11_12.png 404 (Not Found)

---

**Cursor**

The issue is that the **water shader** tries to load individual PNG tiles directly, but we only uploaded the PMTiles file to R2 (which contains all tiles in one archive).

The simplest fix is to disable the water shader in production. Let me update it:

```typescriptreact
import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check if we're in development mode (show debug UI)
const isDev = import.meta.env.DEV;

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = `${__TILES_BASE_URL__}/tiles.pmtiles`;

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch(`${__TILES_BASE_URL__}/tiles/manifest.json`)
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `${__TILES_BASE_URL__}/tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    // Disable water shader in production - it requires individual tile files
    enabled: isDev,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      {isDev && (
        <ControlPanel
          scanlines={scanlines}
          onScanlinesChange={setScanlines}
          waterShader={waterShader}
          onWaterShaderChange={setWaterShader}
        />
      )}

      {isDev && <TileInfo hoveredTile={hoveredTile} viewState={viewState} />}
    </div>
  );
}

export default App;

```

Now rebuild and redeploy:

```bash
uv run python src/app/deploy.py --build --github-pages
```

The water shader is now disabled in production because it needs individual PNG files (which aren't on R2). In dev mode it still works since you have the local tiles.

```shellscript
uv run python src/app/deploy.py --build --github-pages
```

