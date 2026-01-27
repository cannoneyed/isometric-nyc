"""
Download a generation database from R2.

Usage:
  uv run python src/isometric_nyc/download_generation.py tiny-nyc
  uv run python src/isometric_nyc/download_generation.py nyc

Or use MAP_ID from .env:
  uv run python src/isometric_nyc/download_generation.py
"""

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

R2_BASE_URL = "https://isometric-nyc-tiles.cannoneyed.com"


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_size >= 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"
    elif bytes_size >= 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.2f} MB"
    elif bytes_size >= 1024:
        return f"{bytes_size / 1024:.2f} KB"
    else:
        return f"{bytes_size} bytes"


def get_file_size(url: str) -> int | None:
    """Get file size from Content-Length header."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            content_length = response.headers.get("Content-Length")
            if content_length:
                return int(content_length)
    except requests.RequestException:
        pass
    return None


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "wb") as f:
            if total_size > 0:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"  {desc}",
                    ncols=80,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                # No content-length, download without progress
                print(f"  {desc}...")
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        return True
    except requests.RequestException as e:
        print(f"  Error downloading {desc}: {e}")
        return False


def download_generation(generation_id: str) -> int:
    """Download a generation from R2."""
    dest_dir = Path(f"generations/{generation_id}")

    print(f"Downloading {generation_id} generation...")
    print()

    # Check file sizes first
    db_url = f"{R2_BASE_URL}/generations/{generation_id}/quadrants.db"
    config_url = f"{R2_BASE_URL}/generations/{generation_id}/generation_config.json"

    db_size = get_file_size(db_url)
    config_size = get_file_size(config_url)

    print("Files to download:")
    if db_size:
        print(f"  quadrants.db: {format_size(db_size)}")
    else:
        print("  quadrants.db: (checking...)")

    if config_size:
        print(f"  generation_config.json: {format_size(config_size)}")

    print()

    # Download quadrants.db
    db_path = dest_dir / "quadrants.db"
    if not download_file(db_url, db_path, "quadrants.db"):
        print(f"Failed to download quadrants.db")
        return 1

    # Download generation_config.json (optional)
    config_path = dest_dir / "generation_config.json"
    try:
        response = requests.get(config_url, timeout=10)
        if response.status_code == 200:
            config_path.write_bytes(response.content)
            print(f"  generation_config.json: downloaded")
    except requests.RequestException:
        pass  # Config is optional

    print()
    print(f"Done! Generation downloaded to {dest_dir}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a generation database from R2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using MAP_ID from .env
  uv run python src/isometric_nyc/download_generation.py

  # Download a specific generation
  uv run python src/isometric_nyc/download_generation.py tiny-nyc
  uv run python src/isometric_nyc/download_generation.py nyc
        """,
    )
    parser.add_argument(
        "generation_id",
        nargs="?",
        default=None,
        help="Generation ID to download (default: from MAP_ID env var)",
    )

    args = parser.parse_args()

    # Determine generation ID
    generation_id = args.generation_id or os.environ.get("MAP_ID")

    if not generation_id:
        print("Error: No generation ID specified")
        print("  Either pass a generation ID as an argument or set MAP_ID in .env")
        return 1

    return download_generation(generation_id)


if __name__ == "__main__":
    sys.exit(main())
