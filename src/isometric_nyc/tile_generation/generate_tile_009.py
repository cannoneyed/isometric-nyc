import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from isometric_nyc.tile_generation.shared import Images


def generate_tile(tile_dir: str, generation_suffix: str):
  load_dotenv()
  client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

  aspect_ratio = "1:1"

  images = Images(client=client)

  images.add_image_contents(
    id="template",
    path=os.path.join(tile_dir, "template.png"),
    description="IMAGE_INDEX is the MASTER STYLE GUIDE. The LEFT side contains the correct pixel art style. The RIGHT side is white and needs to be drawn.",
  )

  # Reordering: Put whitebox before render to prioritize geometry
  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX is the GEOMETRY BLUEPRINT. This shows the exact 3D shapes of the buildings you must draw.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX is a CORRUPTED, LOW-RES REFERENCE. Use it ONLY for color hints. It is blurry and pixelated - DO NOT COPY ITS TEXTURE.",
  )

  generation_prompt = f"""
**Task:** REPAIR and COMPLETE the isometric city scene. The right side of {images.get_index("template")} is missing.

**The Problem:**
The user has provided a reference image ({images.get_index("render")}) that is **damaged, blurry, and full of noise**.

**Your Goal:**
You must **RESTORE** the scene by re-drawing the buildings shown in the {images.get_index("whitebox")} blueprint, but applying the clean, sharp pixel art style found on the left side of {images.get_index("template")}.

**Critical Instructions:**
1.  **Ignore the Blur:** Do not reproduce the fuzzy, noisy look of {images.get_index("render")}. If your output looks like a resized photo, you have failed.
2.  **Force Sharpness:** Every building must be drawn with **clean, solid pixels**. Imagine you are vectorizing the blurry reference. Use solid colors and hard edges.
3.  **Fix the Seam:**
    *   Look at the streets and rooftops on the left side of {images.get_index("template")}.
    *   **Extend** these lines straight across into the white area.
    *   The new buildings must align perfectly with the existing ones.
4.  **Texture Rules:**
    *   If the blueprint shows a wall, fill it with a clean pattern (brick, concrete, glass) matching the tileset on the left.
    *   Do not leave it blank. Do not make it noisy.

**Summary:**
Trace the {images.get_index("whitebox")}. Color it using hints from {images.get_index("render")}. Style it exactly like the left side of {images.get_index("template")}.
""".strip()

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

  output_path = os.path.join(tile_dir, "generation.png")
  versioned_output_path = os.path.join(tile_dir, f"generation{generation_suffix}.png")
  for part in response.parts:
    if part.text is not None:
      print(part.text)
    elif image := part.as_image():
      print("Saving image...")
      image.save(output_path)
      image.save(versioned_output_path)


def main():
  generation_suffix = os.path.basename(__file__).split(".")[0].split("_")[-1]
  config_path = os.path.join(os.path.dirname(__file__), "config.json")
  with open(config_path, "r") as f:
    config = json.load(f)
  tile_dir = config["tile_dir"]
  generate_tile(tile_dir, generation_suffix)


if __name__ == "__main__":
  main()
