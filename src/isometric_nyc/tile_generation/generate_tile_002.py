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
    description="IMAGE_INDEX is a masked image. The left half contains existing high-quality isometric pixel art. The right half is masked white and needs to be generated.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX is a low-resolution reference for building GEOMETRY and PLACEMENT only. Do NOT use its texture or style.",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX is a depth map showing the 3D structure of the buildings.",
  )

  generation_prompt = f"""
Generate the missing right half of the isometric pixel art image ({images.get_index("template")}) to seamlessly extend the city scene.

**Goal:** Create a high-quality, sharp, isometric pixel art city scene that looks exactly like a classic 90s city builder game (SimCity 2000/3000 style).

**Strict Style Requirements:**
*   **Style:** Pure Pixel Art. Hard edges. 2D Raster graphics.
*   **Perspective:** Fixed Isometric Orthographic projection.
*   **Forbidden:** DO NOT generate photographic textures, blurry details, noise, or realistic lighting. DO NOT copy the visual style of the render ({images.get_index("render")}).
*   **Consistency:** The generated right half MUST match the line width, color palette, lighting angle, and scale of the left half of {images.get_index("template")} EXACTLY.
*   **Seam:** There must be NO visible vertical seam where the left and right halves meet. Streets and buildings must connect perfectly.

**Reference Usage:**
1.  **Geometry:** Use {images.get_index("render")} and {images.get_index("whitebox")} to understand *where* buildings and streets are located and their general shape.
2.  **Appearance:** IGNORE the blurry/noisy appearance of {images.get_index("render")}. Instead, "paint" over that geometry using the clean, sharp pixel art style seen in the left half of {images.get_index("template")}.

**Action:**
Fill in the white masked area on the right side of {images.get_index("template")} with pixel art buildings and streets that correspond to the geometry in the references, but rendered in the strict style of the left side.
""".strip()

  contents = images.contents + [generation_prompt]

  # Debug print
  # for c in contents:
  #   if isinstance(c, str):
  #     print(c)
  #   else:
  #     print(type(c))

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
