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
    description="IMAGE_INDEX is the target canvas. The LEFT half is the style/color reference. The RIGHT half is masked white and must be generated.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX shows building PLACEMENT. Ignore its colors/textures.",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX shows 3D GEOMETRY.",
  )

  generation_prompt = f"""
**Task:** Complete the isometric pixel art city scene in {images.get_index("template")} by filling the white masked area on the right.

**Refined Style & Color Goals:**
1.  **Strict Color Matching:** The user noticed color inconsistencies. You MUST sample your color palette *exclusively* from the existing pixel art on the left side of {images.get_index("template")}. Match the specific shades of brick red, concrete beige, asphalt grey, and window blue used there.
2.  **Architectural Continuity:** The new buildings on the right should look like they are part of the exact same "tileset" as the left. Match the window spacing, cornice details, and roof styles.
3.  **Clean & Consistent:** Maintain the sharp, clean SimCity 3000 aesthetic. No noise, no "dirty" pixels. Uniform flat shading on surfaces.

**Reference Usage:**
*   **Geometry:** Use {images.get_index("render")} and {images.get_index("whitebox")} to know *where* to put buildings and how tall they are.
*   **Style/Color:** IGNORE the colors in {images.get_index("render")}. They are wrong. Use ONLY the colors from the left side of {images.get_index("template")}.

**Instructions:**
*   Extend the city into the white area.
*   Ensure the street level connects perfectly.
*   Texture the buildings using the *exact* materials found on the left (e.g. if the left has red brick rowhouses, use that same red brick texture for similar buildings on the right).
*   Fix any "whitebox" or untextured areas.
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
