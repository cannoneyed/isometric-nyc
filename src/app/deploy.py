#!/usr/bin/env python3
"""
Deployment script for isometric-nyc web app.

This script handles:
1. Building the static web app (JS/HTML/CSS)
2. Pushing the built content to GitHub Pages
3. Publishing DZI tiles to Cloudflare R2 (using rclone)
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
      # Skip tiles and large assets - they'll be served from R2
      if item.name in (
        "tiles",
        "tiles_files",
        "tiles_processed",
        "water_masks",
      ) or item.suffix in (".pmtiles", ".dzi"):
        print(f"    ⏭️  Skipping {item.name} (will be served from R2)")
        continue

      # Skip metadata file (served from R2 with tiles)
      if item.name == "tiles_metadata.json":
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


def upload_to_r2_rclone(files_to_upload: list[tuple[Path, str]], dry_run: bool = False) -> bool:
  """Upload files to R2 using rclone (supports large files via multipart upload)."""
  # Check if rclone is available and configured
  result = subprocess.run(
    ["rclone", "listremotes"],
    capture_output=True,
    text=True,
  )

  if result.returncode != 0:
    return False

  # Check if r2 remote is configured
  if "r2:" not in result.stdout:
    print("  ⚠️  rclone 'r2' remote not configured")
    print("  Configure with: rclone config")
    print("  Use 'Cloudflare R2' as the provider")
    return False

  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      size_mb = local_path.stat().st_size / (1024 * 1024)
      print(f"    ⬆️  Uploading {remote_key} ({size_mb:.1f} MB)...", end=" ", flush=True)

      if dry_run:
        print("(dry run)")
        uploaded += 1
        continue

      # Use rclone copyto for precise destination naming
      result = subprocess.run(
        [
          "rclone",
          "copyto",
          str(local_path),
          f"r2:{R2_BUCKET_NAME}/{remote_key}",
          "--progress",
          "--s3-chunk-size", "50M",  # Use multipart for large files
        ],
        capture_output=True,
        text=True,
      )

      if result.returncode == 0:
        print("✓")
        uploaded += 1
      else:
        print(f"✗")
        if result.stderr:
          print(f"      Error: {result.stderr.strip()}")
        failed += 1

    except Exception as e:
      print(f"✗ ({e})")
      failed += 1

  return failed == 0


def upload_to_r2_wrangler(files_to_upload: list[tuple[Path, str]], dry_run: bool = False) -> bool:
  """Upload files to R2 using wrangler (limited to 300 MiB per file)."""
  uploaded = 0
  failed = 0

  for local_path, remote_key in files_to_upload:
    try:
      size_mb = local_path.stat().st_size / (1024 * 1024)

      # Check file size limit
      if size_mb > 300:
        print(f"    ⚠️  {remote_key} ({size_mb:.1f} MB) exceeds wrangler's 300 MB limit")
        print(f"       Use --use-rclone flag or install rclone for large file support")
        failed += 1
        continue

      print(f"    ⬆️  Uploading {remote_key} ({size_mb:.1f} MB)...", end=" ", flush=True)

      if dry_run:
        print("(dry run)")
        uploaded += 1
        continue

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

  return failed == 0


def upload_dzi_directory(dzi_files_dir: Path, remote_prefix: str, dry_run: bool = False) -> bool:
  """Upload DZI tiles directory to R2 using rclone sync."""
  print(f"    📂 Syncing {dzi_files_dir.name}/ to R2...")

  if dry_run:
    # Count files for dry run
    file_count = sum(1 for _ in dzi_files_dir.rglob("*") if _.is_file())
    print(f"    (dry run) Would sync {file_count} files")
    return True

  # Use rclone sync for efficient directory upload
  result = subprocess.run(
    [
      "rclone",
      "sync",
      str(dzi_files_dir),
      f"r2:{R2_BUCKET_NAME}/{remote_prefix}",
      "--progress",
      "--transfers", "16",  # Parallel transfers
      "--checkers", "32",   # Parallel checkers
      "--fast-list",        # Use fewer API calls
    ],
    capture_output=False,  # Show progress
  )

  return result.returncode == 0


def upload_to_r2(export_dir: str = "dzi", dry_run: bool = False) -> bool:
  """Upload DZI tiles and metadata to Cloudflare R2.

  Args:
    export_dir: Name of the export directory inside public/ (default: "dzi")
    dry_run: If True, show what would be done without making changes
  """
  print(f"\n☁️  Uploading '{export_dir}/' to Cloudflare R2...")

  # DZI requires rclone for efficient directory sync
  has_rclone = shutil.which("rclone") is not None

  if not has_rclone:
    print("  ❌ Error: rclone is required for DZI upload (many small files)")
    print("  Install rclone: brew install rclone")
    print("  Then configure: rclone config (use 'r2' as remote name, 'Cloudflare R2' provider)")
    return False

  # Check if r2 remote is configured
  result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
  if "r2:" not in result.stdout:
    print("  ❌ Error: rclone 'r2' remote not configured")
    print("  Configure with: rclone config")
    print("  Use 'Cloudflare R2' as the provider")
    return False

  # Find DZI files in the export directory
  export_path = PUBLIC_DIR / export_dir
  dzi_file = export_path / "tiles.dzi"
  dzi_files_dir = export_path / "tiles_files"
  metadata_file = export_path / "metadata.json"

  if not export_path.exists():
    print(f"  ❌ Error: Export directory not found: {export_path}")
    print("  Run the DZI export first:")
    print(f"    uv run python src/isometric_nyc/generation/export_dzi.py generations/nyc --export-dir {export_dir}")
    return False

  if not dzi_file.exists():
    print(f"  ❌ Error: tiles.dzi not found in {export_path}")
    return False

  if not dzi_files_dir.exists():
    print(f"  ❌ Error: tiles_files/ directory not found in {export_path}")
    return False

  success = True

  # Upload single files (DZI descriptor and metadata)
  # Remote structure: r2:bucket/{export_dir}/tiles.dzi, etc.
  single_files: list[tuple[Path, str]] = [
    (dzi_file, f"{export_dir}/tiles.dzi"),
  ]

  if metadata_file.exists():
    single_files.append((metadata_file, f"{export_dir}/metadata.json"))

  print(f"  📊 Uploading DZI files to R2 ({export_dir}/)...")

  # Upload single files
  for local_path, remote_key in single_files:
    size_kb = local_path.stat().st_size / 1024
    print(f"    ⬆️  Uploading {local_path.name} ({size_kb:.1f} KB)...", end=" ", flush=True)

    if dry_run:
      print("(dry run)")
      continue

    result = subprocess.run(
      [
        "rclone",
        "copyto",
        str(local_path),
        f"r2:{R2_BUCKET_NAME}/{remote_key}",
      ],
      capture_output=True,
      text=True,
    )

    if result.returncode == 0:
      print("✓")
    else:
      print(f"✗")
      if result.stderr:
        print(f"      Error: {result.stderr.strip()}")
      success = False

  # Upload tiles directory using sync
  if dzi_files_dir.exists():
    # Count total files and size
    total_files = sum(1 for _ in dzi_files_dir.rglob("*") if _.is_file())
    total_size_mb = sum(f.stat().st_size for f in dzi_files_dir.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"    📂 Syncing tiles_files/ ({total_files} files, {total_size_mb:.0f} MB)...")

    if not upload_dzi_directory(dzi_files_dir, f"{export_dir}/tiles_files", dry_run):
      success = False

  if success:
    print("  ✅ R2 upload complete!")
    print(f"  🔗 DZI tiles available at: https://isometric-nyc-tiles.cannoneyed.com/{export_dir}/")
    print(f"     - {export_dir}/tiles.dzi")
    print(f"     - {export_dir}/metadata.json")
    print(f"     - {export_dir}/tiles_files/")
  else:
    print("  ⚠️  Some uploads failed")

  return success


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Deploy isometric-nyc web app",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Prerequisites:
  - bun: Install with `curl -fsSL https://bun.sh/install | bash`
  - rclone: Install with `brew install rclone` (required for DZI tile upload)

Configuring rclone for R2:
  1. Run `rclone config`
  2. Create new remote named 'r2'
  3. Choose 'Cloudflare R2' as provider
  4. Enter your R2 access key and secret
  5. Set endpoint to your R2 endpoint URL

Environment Variables (optional):
  GITHUB_PAGES_DOMAIN   Custom domain for GitHub Pages

Tile Format:
  This script uploads DZI (Deep Zoom Image) tiles to R2:
  - tiles.dzi            DZI descriptor file
  - tiles_metadata.json  Custom metadata (grid dimensions, origin, etc.)
  - tiles_files/         Tile pyramid directory (many small WebP files)

  Generate DZI tiles with:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib uv run python \\
      src/isometric_nyc/generation/export_dzi.py generations/nyc

Examples:
  # Build only
  python deploy.py --build

  # Full deployment (build + GitHub Pages + R2)
  python deploy.py --all

  # Dry run to see what would be deployed
  python deploy.py --all --dry-run

  # Just upload DZI tiles to R2
  python deploy.py --r2

  # Deploy only web assets (build + GitHub Pages, no tile uploads)
  python deploy.py --web-only
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
    "--export-dir",
    type=str,
    default="dzi",
    help="Export directory to upload (default: dzi)",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Run all deployment steps",
  )
  parser.add_argument(
    "--web-only",
    action="store_true",
    help="Build and deploy web assets only (no tile uploads)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )
  args = parser.parse_args()

  # If no options specified, show help
  if not any([args.build, args.github_pages, args.r2, args.all, args.web_only]):
    parser.print_help()
    return 0

  print("🚀 Isometric NYC Deployment")
  print("=" * 40)

  success = True

  # Build
  if args.build or args.all or args.web_only:
    if not build_app():
      success = False
      if not args.all and not args.web_only:
        return 1

  # GitHub Pages
  if args.github_pages or args.all or args.web_only:
    if not deploy_github_pages(dry_run=args.dry_run):
      success = False
      if not args.all and not args.web_only:
        return 1

  # R2 upload (skip for --web-only)
  if args.r2 or args.all:
    if not upload_to_r2(export_dir=args.export_dir, dry_run=args.dry_run):
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
