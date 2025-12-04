import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from isometric_nyc.tile_generation.shared import Images

REFERENCE_IMAGE_NAME = "style_a.png"


def pixelate_image(input_path, output_path):
  try:
    # Open the image
    with Image.open(input_path) as img:
      print(f"Original size: {img.size}")

      # 1. Downscale to 512x512
      # We use BILINEAR here to blend the messy AI artifacts together
      img_small = img.resize((512, 512), resample=Image.Resampling.BILINEAR)

      # 2. Upscale back to 1024x1024
      # We use NEAREST here to lock the pixels into a hard grid without blurring
      img_pixelated = img_small.resize((1024, 1024), resample=Image.Resampling.NEAREST)

      # Save the result
      img_pixelated.save(output_path)
      print(f"Saved pixelated image to: {output_path}")

  except Exception as e:
    print(f"Error: {e}")


def postprocess_generation(tile_dir: str, reference_image_name: Path):
  load_dotenv()
  client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

  aspect_ratio = "1:1"

  view_json_path = tile_dir / "view.json"
  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  references_dir = Path("references")
  reference_path = references_dir / reference_image_name
  if not reference_path.exists():
    raise FileNotFoundError(f"{reference_image_name} not found in {references_dir}")

  images = Images(client=client)

  images.add_image_contents(
    id="generation",
    path=os.path.join(tile_dir, "generation.png"),
    description="IMAGE_INDEX shows the isometric pixel art generation.",
  )

  images.add_image_contents(
    id="reference",
    path=reference_path,
    description="IMAGE_INDEX is the style reference for the isometric pixel art. Follow this style exactly*. Do not deviate from it.",
  )

  generation_prompt = f"""
**Task:** Isometric pixel art city scene in the style of {images.get_index("reference")}. 16-bit pixel art, SimCity 2000 style, isometric strategy game asset, flat shading, clean vector lines, vivid solid colors, hard edges, sharp focus, aliased. Quantized colors, limited palette.

Please modify {images.get_index("generation")} to be more in the style of {images.get_index("reference")}, **reducing the level of detail** from the original generation and ensuring that there are more crisp grid pixels. Ensure that windows, ledges, and roof details are present and correct.

Avoid: noise, texture, dithering, gradient, blur, bokeh, realistic, dirty, painting, oil painting, brushstrokes, ambient occlusion.
""".strip()

  if view_json.get("special_instructions"):
    generation_prompt += (
      f"\n**Special Instructions:**\n{view_json['special_instructions']}"
    )

  contents = images.contents + [generation_prompt]

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

  output_path = os.path.join(tile_dir, "generation_post.png")
  for part in response.parts:
    if part.text is not None:
      print(part.text)
    elif image := part.as_image():
      print("Saving image...")
      image.save(output_path)


def main():
  parser = argparse.ArgumentParser(
    description="Post-process the generation of isometric pixel art for a tile."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, whitebox.png, render.png)",
  )
  parser.add_argument(
    "--reference_image_name",
    default="style_a.png",
    help="Directory containing reference images (simcity.jpg)",
  )

  args = parser.parse_args()

  tile_dir = Path(args.tile_dir)
  reference_image_name = args.reference_image_name
  postprocess_generation(tile_dir, reference_image_name)


if __name__ == "__main__":
  main()
