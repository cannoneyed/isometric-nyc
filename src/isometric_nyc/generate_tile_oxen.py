"""
Generate tiles using the Oxen.ai fine-tuned model for infill generation.

This script creates an infill image by compositing neighboring tile generations
with the current tile's render, uploads it to Google Cloud Storage, and uses
the Oxen API to generate the pixel art infill.

Usage:
  uv run python src/isometric_nyc/generate_tile_oxen.py --tile_dir <path_to_tile>
"""

import argparse
import os
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from google.cloud import storage


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Prerequisites:
  1. Create a GCS bucket with public access enabled
  2. Set up authentication via GOOGLE_APPLICATION_CREDENTIALS environment variable
     or use default credentials (gcloud auth application-default login)

  Args:
      local_path: Path to the local file to upload
      bucket_name: Name of the GCS bucket
      blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
      Public URL of the uploaded file
  """
  # Initialize the GCS client
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  # Generate a unique blob name if not provided
  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  # Upload the file
  print(f"Uploading {local_path} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))

  # Make the blob publicly accessible
  blob.make_public()

  public_url = blob.public_url
  print(f"File uploaded successfully. Public URL: {public_url}")

  return public_url


def call_oxen_api(image_url: str, api_key: str) -> str:
  """
  Call the Oxen API to generate an infill image.

  Args:
      image_url: Public URL of the infill image
      api_key: Oxen API key

  Returns:
      URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-odd-blue-marmot",  # V04 generation model
    "input_image": image_url,
    "prompt": "Convert the input image to <isometric nyc pixel art>",
    "num_inference_steps": 28,
  }

  print(f"Calling Oxen API with image: {image_url}")
  print(f"Prompt: {payload['prompt']}")

  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()

  result = response.json()
  print(f"Oxen API response: {result}")

  # The API returns images in: {"images": [{"url": "..."}], ...}
  if "images" in result and len(result["images"]) > 0:
    return result["images"][0]["url"]
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  else:
    raise ValueError(f"Unexpected API response format: {result}")


def download_image(url: str, output_path: Path) -> None:
  """
  Download an image from a URL and save it to the specified path.

  Args:
      url: URL of the image to download
      output_path: Path where the image should be saved
  """
  print(f"Downloading generated image from {url}...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)

  print(f"Image saved to {output_path}")


def generate_tile(tile_dir: str, bucket_name: str) -> None:
  """
  Generate a tile using the Oxen API.

  Args:
      tile_dir: Path to the tile directory
      bucket_name: GCS bucket name for uploading images
  """
  load_dotenv()

  # Get API key from environment
  api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
  if not api_key:
    raise ValueError(
      "OXEN_INFILL_V02_API_KEY environment variable not set. "
      "Please add it to your .env file."
    )

  tile_dir_path = Path(tile_dir)

  # Step 1: Get the render image
  render_path = tile_dir_path / "render.png"

  # Step 2: Upload to Google Cloud Storage
  print("\n" + "=" * 60)
  print("STEP 2: Uploading to Google Cloud Storage")
  print("=" * 60)
  image_url = upload_to_gcs(render_path, bucket_name)

  # Step 3: Call Oxen API
  print("\n" + "=" * 60)
  print("STEP 3: Calling Oxen API")
  print("=" * 60)
  generated_url = call_oxen_api(image_url, api_key)

  # Step 4: Download the result
  print("\n" + "=" * 60)
  print("STEP 4: Downloading generated image")
  print("=" * 60)
  output_path = tile_dir_path / "generation.png"
  download_image(generated_url, output_path)

  print("\n" + "=" * 60)
  print("GENERATION COMPLETE!")
  print(f"Output saved to: {output_path}")
  print("=" * 60)


def main():
  parser = argparse.ArgumentParser(
    description="Generate isometric pixel art tiles using Oxen API."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, render.png)",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="Google Cloud Storage bucket name for uploading images",
  )

  args = parser.parse_args()
  generate_tile(args.tile_dir, args.bucket)


if __name__ == "__main__":
  main()
