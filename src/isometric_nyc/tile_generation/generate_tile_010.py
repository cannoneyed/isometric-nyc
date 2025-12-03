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

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX is the GEOMETRY BLUEPRINT. Use this for exact shapes and positions.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX is the MATERIAL REFERENCE. Use this to determine if a building is beige stone, red brick, or blue glass.",
  )

  generation_prompt = f"""
**Task:** COMPLETE the isometric city scene by drawing the missing right half of {images.get_index("template")}.

**Workflow:**
1.  **Geometry:** Trace the shapes from the {images.get_index("whitebox")}. Extend the street grid from the left side of {images.get_index("template")} seamlessly.
2.  **Materials:** Look at {images.get_index("render")} to decide **WHAT** to draw (e.g., "This tall building is beige stone", "This small building is red brick").
3.  **Style:** Look at the left side of {images.get_index("template")} to decide **HOW** to draw it (e.g., "Draw beige stone using this specific pixel pattern", "Draw windows using this crisp style").

**Crucial Correction:**
*   **Do not ignore the render's colors:** In the previous attempt, you drew a grey modern building where a beige Art Deco skyscraper exists. You MUST look at {images.get_index("render")} to get the correct color and architectural type (e.g. Art Deco vs Glass Tower).
*   **Do not copy the render's blur:** While you use the *color*, you must still use the *clean pixel lines* of the {images.get_index("template")}.

**Summary:**
Correct Color/Material (from Render) + Correct Shape (from Whitebox) + Correct Pixel Style (from Template) = Success.
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
