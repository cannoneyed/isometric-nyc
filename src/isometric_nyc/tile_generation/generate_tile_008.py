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
    description="IMAGE_INDEX is the MASTER STYLE GUIDE. The left side is the target quality.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX is a ROUGH COLOR SKETCH. Use it ONLY for knowing if a building is brick/glass/concrete. DO NOT USE ITS PIXELS.",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX is the BLUEPRINT. This is the primary source for SHAPE and GEOMETRY.",
  )

  generation_prompt = f"""
**Task:** You are an expert Pixel Artist. Your task is to DRAW new game assets into the white masked area of {images.get_index("template")}.

**Workflow - "Blueprint to Sprite":**
1.  **Look at the Blueprint ({images.get_index("whitebox")}):** This image shows you the exact shape, height, and position of every building. Trace these shapes internally. This is your source of truth for *geometry*.
2.  **Look at the Color Sketch ({images.get_index("render")}):** Glance at this ONLY to see what material a building is made of (e.g., "Oh, that tall one is beige stone", "That small one is red brick"). **IGNORE THE BLUR.** Do not copy the pixels.
3.  **Apply the Style ({images.get_index("template")}):** Using the palette and technique from the left side of the template, **DRAW** the buildings defined by the Blueprint.

**Rules for Success:**
*   **NO FILTERING:** Do not just apply a filter to the {images.get_index("render")}. It is low quality. You must DRAW new pixels from scratch based on the {images.get_index("whitebox")}.
*   **SHARP LINES:** Since you are tracing the clean {images.get_index("whitebox")}, your output lines must be razor sharp. 2:1 isometric slopes.
*   **CONSISTENT DETAIL:** Every surface from the Blueprint must be filled with pixel art texture (windows, bricks) matching the Style Guide.

**Correction Mechanism:**
*   If you see a blurry blob in your output -> You looked at the Render too much. **Look at the Whitebox instead.**
*   If you see a missing building -> You ignored the Whitebox. **Trace the Whitebox.**
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
