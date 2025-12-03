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
    description="IMAGE_INDEX is the target canvas. The LEFT half is the style reference (perfect isometric pixel art). The RIGHT half is masked white and must be generated.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX shows the building PLACEMENT and VOLUME. Ignore its blurry texture.",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX shows the 3D GEOMETRY. Ignore its lack of color.",
  )

  generation_prompt = f"""
**Task:** Complete the isometric pixel art city scene in {images.get_index("template")} by filling the white masked area on the right.

**Critical Style Rules (SimCity 3000 Aesthetic):**
1.  **Clean Pixel Art:** The output must be sharp, clean, and detailed. Avoid noise, random dithering, or messy "high pixelation" artifacts. The pixels should represent defined windows, ledges, and roof details.
2.  **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc., matching the palette of the left side. Do not leave any geometry blank or white.
3.  **Seamless Integration:** The generated right half must connect *perfectly* to the left half. Streets, rooflines, and building facades must align without breaks, fragmentation, or visible seams.
4.  **Ignore Render Artifacts:** Use {images.get_index("render")} ONLY for position. Do NOT copy its blurry, photo-realistic, or noisy look. The result should look like a hand-drawn game sprite, not a downscaled photo.

**Instructions:**
*   Look at the buildings on the left of {images.get_index("template")}. Note their window patterns, roof colors, and shadow angles.
*   Extend this exact style into the white area.
*   Use {images.get_index("whitebox")} to determine the height and shape of the new buildings.
*   Ensure straight, clean isometric lines (2:1 pixel slope).
*   **Correction:** If you see a "whitebox" building in your internal preview, TEXTURE IT. If you see noise, CLEAN IT UP.

**Negative Constraints:**
*   NO photographic textures.
*   NO blurry upscaling artifacts.
*   NO untextured white/grey blocks.
*   NO mismatched shadows (light comes from the same direction as the left side).
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
