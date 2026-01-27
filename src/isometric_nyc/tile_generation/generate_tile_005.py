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
    description="IMAGE_INDEX is the target canvas. The LEFT half is the clean, sharp style reference. The RIGHT half is masked white and must be generated.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX shows building PLACEMENT. Ignore its blurry texture.",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX shows 3D GEOMETRY.",
  )

  generation_prompt = f"""
**Task:** Complete the isometric pixel art city scene in {images.get_index("template")} by filling the white masked area on the right.

**Refined Style Goals (Crisp & Clean):**
1.  **Eliminate Blur & Fuzziness:** The user reported "blurry/pixelated" buildings. We need **RAZOR SHARP** edges. Do not produce fuzzy lines, anti-aliasing, or upscaling artifacts.
2.  **Pixel-Perfect Definition:** Every window, ledge, and cornice must be drawn with precise, hard pixels. Think of this as a high-resolution sprite, not a low-res upscale.
3.  **Solid Colors over Noise:** Avoid noisy textures (like scattered random pixels). Use solid blocks of color with clean shading to define forms. If a wall is brick, draw the brick pattern cleanly or use a solid color with a texture hint, not a noisy mess.
4.  **Consistent Polishing:** All buildings, even small ones, must receive the same level of detail and sharpness as the main structures on the left.

**Reference Usage:**
*   **Style:** Match the left side of {images.get_index("template")} exactly. Copy its clean, "vector-like" pixel art quality.
*   **Geometry:** Use {images.get_index("render")} and {images.get_index("whitebox")} ONLY for shape and position.

**Instructions:**
*   "Paint" the missing buildings into the white area.
*   Use the palette from the left side.
*   **Force Sharpness:** If a building looks blurry or like a bad JPEG in your preview, REDRAW it with hard edges.
*   Ensure the street grid connects perfectly without any visual breaks.

**Negative Constraints:**
*   NO blur.
*   NO anti-aliasing on outer edges.
*   NO "fuzzy" details.
*   NO JPEG compression artifacts.
*   NO noisy/random pixel clusters.
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
