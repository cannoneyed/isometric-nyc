"""
Upload a generations directory to Cloudflare R2 for public download.

This script uploads the quadrants.db and generation_config.json from a
generations directory to R2, making them available for download.

Usage:
  uv run python src/isometric_nyc/upload_generation_to_r2.py tiny-nyc

Requirements:
  - rclone configured with an 'r2' remote pointing to your R2 bucket
"""

import argparse
import subprocess
import sys
from pathlib import Path

R2_BUCKET = "isometric-nyc"
R2_REMOTE = "r2"


def upload_generation(generation_dir: Path) -> None:
  """Upload a generation directory to R2."""
  generation_id = f"generations/{generation_dir.name}"

  # Files to upload
  files_to_upload = [
    # ("quadrants.db", "quadrants.db"),
    ("generation_config.json", "generation_config.json"),
  ]

  r2_path = f"generations/{generation_id}"

  for local_name, remote_name in files_to_upload:
    local_path = generation_dir / local_name
    if not local_path.exists():
      print(f"  Skipping {local_name} (not found)")
      continue

    remote_path = f"{R2_REMOTE}:{R2_BUCKET}/{r2_path}/{remote_name}"
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  Uploading {local_name} ({size_mb:.1f} MB)...")

    result = subprocess.run(
      ["rclone", "copyto", str(local_path), remote_path, "--progress"],
      check=False,
    )

    if result.returncode != 0:
      print(f"  Failed to upload {local_name}")
      sys.exit(1)

  print()
  print("Upload complete!")
  print()
  print("Download commands for users:")
  print("-" * 60)
  print(f"# Download {generation_id} database:")
  print(f"curl -L -o generations/{generation_id}/quadrants.db \\")
  print(f"  'https://isometric-nyc-tiles.cannoneyed.com/{r2_path}/quadrants.db'")
  print()
  print("# Or with wget:")
  print(f"wget -O generations/{generation_id}/quadrants.db \\")
  print(f"  'https://isometric-nyc-tiles.cannoneyed.com/{r2_path}/quadrants.db'")


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Upload a generations directory to R2.",
  )
  parser.add_argument(
    "map_id",
    type=str,
    help="Map ID to upload (e.g., tiny-nyc). Loads from generations/<map_id>/",
  )

  args = parser.parse_args()

  generation_dir = (Path("generations") / args.map_id).resolve()

  if not generation_dir.exists():
    print(f"Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"Error: Not a directory: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"Error: No quadrants.db found in {generation_dir}")
    return 1

  print(f"Uploading {args.map_id} to R2...")
  upload_generation(generation_dir)
  return 0


if __name__ == "__main__":
  sys.exit(main())
