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
