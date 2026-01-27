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
    description="IMAGE_INDEX shows building PLACEMENT and primary features (like prominent window grids). Ignore its blurry texture.",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX shows 3D GEOMETRY.",
  )

  generation_prompt = f"""
**Task:** Complete the isometric pixel art city scene in {images.get_index("template")} by filling the white masked area on the right.

**CRITICAL TEXTURE & DETAIL RULES (SimCity 3000 Aesthetic):**
1.  **NO BLANK SURFACES:** Every building surface **must** have pixel art texture and detail (windows, doors, vents, roof patterns, etc.). Absolutely no plain, untextured blocks or white facades.
2.  **INFER DETAIL:** If the {images.get_index("render")} provides vague or no specific detail for a building surface, you **must invent appropriate pixel art detail** (e.g., standard window grids, consistent brick patterns, roof features) that aligns perfectly with the style, density, and detail level of the existing pixel art on the left side of {images.get_index("template")}.
3.  **RAZOR SHARP LINES:** Maintain all previous instructions for crisp, clean, pixel-perfect edges and lines. Avoid any blur, fuzziness, anti-aliasing, or noisy pixel artifacts.

**Reference Usage:**
*   **Style:** Match the left side of {images.get_index("template")} exactly. Copy its clean, "vector-like" pixel art quality, including color palette and shadow angles.
*   **Geometry:** Use {images.get_index("render")} and {images.get_index("whitebox")} ONLY for building shape, position, and the general layout of prominent features like window grids. Do NOT copy their colors or blurry textures.

**Instructions:**
*   "Paint" the missing buildings into the white area.
*   Ensure the street grid connects perfectly without any visual breaks.
*   **Self-Correction:** If any building appears blank, untextured, or blurry in your generated output, refine it to include pixel art detail with sharp edges.
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
