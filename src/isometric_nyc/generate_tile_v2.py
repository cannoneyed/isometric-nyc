import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from isometric_nyc.tile_generation.shared import Images

REFERENCE_IMAGE_NAME = "style_a.png"


def generate_tile(tile_dir: str, reference_image_name: Path):
  load_dotenv()
  client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

  aspect_ratio = "1:1"

  view_json_path = tile_dir / "view.json"
  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  references_dir = Path("references")
  reference_path = references_dir / reference_image_name
  if not reference_path.exists():
    raise FileNotFoundError(f"{reference_image_name} not found in {references_dir}")

  images = Images(client=client)

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render.png"),
    description="IMAGE_INDEX shows the building PLACEMENT and VOLUME. Ignore its blurry texture.",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX shows the 3D GEOMETRY. Ignore its lack of color.",
  )

  images.add_image_contents(
    id="reference",
    path=reference_path,
    description="IMAGE_INDEX is the style reference for the isometric pixel art. Follow this style exactly*. Do not deviate from it.",
  )

  generation_prompt = f"""
**Task:** Isometric pixel art city scene in the style of {images.get_index("reference")}.

{images.get_index("render")} is the 3D render of the city - use this image as a reference for the details, textures, colors, and lighting of the buildings, but DO NOT  just use these pixels - we want to copy these details but use the style of {images.get_index("reference")}.

Use the guides in {images.get_index("whitebox")} as the blueprint for all building shapes and locations. Check carefully to make sure every building in {images.get_index("render")} and {images.get_index("whitebox")} is present in the generation, and ensure that the colors and textures of the buildings are correct. If a building is present in {images.get_index("render")} but not in {images.get_index("whitebox")}, it MUST BE generated.

**Critical Style Rules (SimCity 3000 Aesthetic):**
1.  **Clean Pixel Art:** The output must be clean, and detailed but low-resolution, as if on a VGA monitor. Avoid noise, random dithering, or messy "high pixelation" artifacts. The pixels should represent defined windows, ledges, and roof details.
2.  **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc., matching the palette of {images.get_index("reference")}. Do not leave any geometry blank or white.
4.  **Ignore Render Artifacts:** Use {images.get_index("render")} ONLY for position. Do NOT copy its blurry, photo-realistic, or noisy look. The result should look like a hand-drawn game sprite, not a downscaled photo.

**Instructions:**
*   Use {images.get_index("whitebox")} and {images.get_index("render")} to determine the height and shape of the new buildings.
*   Ensure straight, clean isometric lines (2:1 pixel slope).
*   **Correction:** If you see a "whitebox" building in your internal preview, TEXTURE IT. If you see noise, CLEAN IT UP.

**Style Instructions:**
(((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

**Negative Constraints:**
*   NO photographic textures.
*   NO blurry upscaling artifacts.
*   NO untextured white/grey blocks.
*   NO mismatched shadows (light comes from the same direction as {images.get_index("reference")}).
""".strip()

  if view_json.get("special_instructions"):
    generation_prompt += (
      f"\n**Special Instructions:**\n{view_json['special_instructions']}"
    )

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
  for part in response.parts:
    if part.text is not None:
      print(part.text)
    elif image := part.as_image():
      print("Saving image...")
      image.save(output_path)


def main():
  parser = argparse.ArgumentParser(
    description="Generate isometric pixel art for a tile."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, whitebox.png, render.png)",
  )
  parser.add_argument(
    "--reference_image_name",
    default="style_a.png",
    help="Directory containing reference images (simcity.jpg)",
  )

  args = parser.parse_args()

  tile_dir = Path(args.tile_dir)
  reference_image_name = args.reference_image_name
  generate_tile(tile_dir, reference_image_name)


if __name__ == "__main__":
  main()
