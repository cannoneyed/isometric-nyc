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
    description="IMAGE_INDEX is the GEOMETRY BLUEPRINT. Use this for exact shapes and positions. This image is the ABSOLUTE TRUTH for building existence.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX is the MATERIAL REFERENCE. Use this to determine the material, texture, and color of the buildings.",
  )

  generation_prompt = f"""
**Task:** COMPLETE the isometric city scene by drawing the missing right half of {images.get_index("template")}.

**Workflow:**
1.  **Geometry (The Law):** Trace the shapes from the {images.get_index("whitebox")}.
    *   **STRICT RULE:** {images.get_index("whitebox")} is the boss. If it shows a building (grey/white volume), you **MUST** draw a building. If it shows ground (black), you draw ground.
    *   **NO HALLUCINATIONS:** Do not turn a building volume into a flat park. Do not put a building on flat ground.
2.  **Materials:** Look at {images.get_index("render")} to decide **WHAT** to draw.
    *   **Detail Check:** Look closely at roofs. If the render shows cooling towers, vents, or specific structures, draw them. Do not simplify complex roofs into flat noise.
    *   **Color:** Match the facade colors (beige, red brick, glass) from the render.
3.  **Style:** Look at the left side of {images.get_index("template")} to decide **HOW** to draw it (e.g., "Draw beige stone using this specific pixel pattern").

**Crucial Corrections from previous run:**
*   **Fix the Park Hallucination:** In the last run, you drew a park where the whitebox clearly showed a building volume. Follow the whitebox volume strictly.
*   **Fix the Cooling Towers:** The main building has circular cooling towers on the roof. Draw them clearly.

**Summary:**
Whitebox = Shape (Don't change it). Render = Color/Detail (Don't ignore it). Template = Style.
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
