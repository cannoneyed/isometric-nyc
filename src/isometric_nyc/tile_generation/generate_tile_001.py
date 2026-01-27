import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from isometric_nyc.tile_generation.shared import Images


def generate_tile(tile_dir: str, generation_suffix: str):
  load_dotenv()
  client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

  aspect_ratio = "1:1"  # "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"

  images = Images(client=client)

  images.add_image_contents(
    id="template",
    path=os.path.join(tile_dir, "template.png"),
    description="IMAGE_INDEX is a masked image that contains parts of neighboring tiles that have already been generated, with a portion of the image masked out in white.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render_256.png"),
    description="IMAGE_INDEX is a rendered view of the 3D building data using Google 3D tiles API at 256x256 resolution",
  )

  images.add_image_contents(
    id="whitebox",
    path=os.path.join(tile_dir, "whitebox.png"),
    description="IMAGE_INDEX is a depth map geometry of isometric render of a section of New York City.",
  )

  generation_prompt = f"""
Generate the missing half of an isometric pixel art image of a small section of New York City, in the style of classic computer city builder games such as SimCity 3000.

Image Descriptions:
{images.get_descriptions()}

Style Instructions:
(((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

Generation Instructions:
{images.get_index("template")} is has the right half covered in white. Please replace the white part of the masked image with the missing isometric pixel art generation. The buildings must flow seamlessly across the scene. The missing half of the masked image has pixel art buildings that correspond the 3D building data in ({images.get_index("render")}), however the style of the generated image must **exactly** match the style of the existing parts of the masked image. There must be no gaps or inconsistencies between the existing parts of the template and the generated parts of the generated image.

DO NOT CHANGE THE EXISTING PARTS OF {images.get_index("template")}!!!
""".strip()

  contents = images.contents + [generation_prompt]

  for c in contents:
    if isinstance(c, str):
      print(c)
    else:
      print(type(c))

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
