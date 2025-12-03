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
    description="IMAGE_INDEX is the MASTER STYLE REFERENCE. The left half is perfect. The right half is white and must be drawn.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX is a LOW-QUALITY placeholder showing building VOLUMES. It is noisy and ugly. DO NOT COPY ITS LOOK.",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX is a GEOMETRY guide showing 3D shapes.",
  )

  generation_prompt = f"""
**Task:** Create a high-quality isometric pixel art scene in the white masked area of {images.get_index("template")}.

**CRITICAL INSTRUCTION: IGNORE THE RENDER'S TEXTURE**
The reference image {images.get_index("render")} contains "ugly", noisy, low-resolution satellite textures.
*   **DO NOT** copy these textures.
*   **DO NOT** create a "pixelated photo" look.
*   **DO NOT** let the blurry details of the render bleed into your output.

**INSTEAD: APPLY THE STYLE FROM THE LEFT**
You must act as a concept artist who takes the *shapes* from {images.get_index("render")} and **REDRAWS** them using the beautiful, sharp pixel art style found on the left side of {images.get_index("template")}.

**Style Rules:**
1.  **Vector-Sharp Edges:** Every building edge must be a clean, hard line. No fuzziness.
2.  **Hand-Drawn Aesthetic:** The windows, bricks, and roofs should look like they were drawn by a pixel artist, not a filter applied to a photo.
3.  **Invent Clean Details:** If the render shows a blurry gray wall, YOU must draw a sharp, clean wall with a specific texture (e.g., distinct brick pattern or clean concrete panels) that matches the "tileset" of the left side.
4.  **Seamless Connection:** The new drawing must connect perfectly to the existing art on the left.

**Action:**
Fill the white space. Look at the *shapes* in the render, but paint them with the *style* of the template.
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
