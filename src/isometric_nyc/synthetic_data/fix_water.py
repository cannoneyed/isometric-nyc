import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from isometric_nyc.tile_generation.shared import Images


def fix_water(tile_dir: Path):
  load_dotenv()
  client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

  aspect_ratio = "1:1"

  view_json_path = tile_dir / "view.json"
  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  # Assuming references are in the root references/ directory
  project_root = Path(__file__).resolve().parents[3]
  references_dir = project_root / "references"

  images = Images(client=client)

  images.add_image_contents(
    id="generation",
    path=os.path.join(tile_dir, "generation.png"),
    description="IMAGE_INDEX is the isometric pixel art generation to be fixed",
  )

  # Add tree references
  water_dir = references_dir / "water"
  tree_reference_names = [f.name for f in water_dir.glob("*.png")]

  print(f"Found {len(tree_reference_names)} tree reference images in {water_dir}")
  print(f"Tree reference images: {tree_reference_names}")

  tree_ref_ids = []
  for i, ref_name in enumerate(tree_reference_names):
    ref_path = water_dir / ref_name
    if not ref_path.exists():
      print(f"Warning: Reference {ref_name} not found in {water_dir}")
      continue

    ref_id = f"tree_ref_{i}"
    images.add_image_contents(
      id=ref_id,
      path=ref_path,
      description="IMAGE_INDEX is a TARGET STYLE reference specifically for the TREES.",
    )
    tree_ref_ids.append(ref_id)

  generation_prompt = f"""
**Task:** Isometric pixel art city scene - Tree Style Fix

Edit the water pixels in {images.get_index("generation")} to match the style of the other reference images.

Please remove extra details from the water pixels in {images.get_index("generation")} - **reducing the level of detail** from the original generation and ensuring that there are more crisp grid pixels. There should be no grid lines or lines in the water.

EVERYTHING ELSE MUST REMAIN THE SAME. DO NOT ADD ANY NEW TREES. Only fix the existing water pixels in {images.get_index("generation")}.

Avoid: noise, texture, dithering, gradient, blur, bokeh, realistic, dirty, painting, oil painting, brushstrokes, ambient occlusion.
""".strip()

  if view_json.get("special_instructions"):
    generation_prompt += (
      f"\n**Special Instructions:**\n{view_json['special_instructions']}"
    )

  contents = images.contents + [generation_prompt]

  print("Generating content...")
  response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio=aspect_ratio,
      ),
    ),
  )

  output_path = os.path.join(tile_dir, "generation_fixed.png")
  for part in response.parts:
    if part.text is not None:
      print(part.text)
    elif image := part.as_image():
      print(f"Saving fixed generation to {output_path}...")
      image.save(output_path)


def main():
  parser = argparse.ArgumentParser(
    description="Regenerate isometric pixel art tile with specific water styling."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, whitebox.png, render.png)",
  )

  args = parser.parse_args()

  tile_dir = Path(args.tile_dir)
  fix_water(tile_dir)


if __name__ == "__main__":
  main()
