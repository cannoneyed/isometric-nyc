import json
import os
from typing import List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

from isometric_nyc.tile_generation.shared import Images


def check_generation(tile_dir: str):
  load_dotenv()
  client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

  images = Images(client=client)

  images.add_image_contents(
    id="template",
    path=os.path.join(tile_dir, "template.png"),
    description="IMAGE_INDEX is a masked image that contains parts of neighboring tiles that have already been generated, with a portion of the image masked out in white.",
  )

  images.add_image_contents(
    id="render",
    path=os.path.join(tile_dir, "render.png"),
    description="IMAGE_INDEX is a rendered view of the 3D building data using Google 3D tiles API",
  )

  images.add_image_contents(
    id="generation",
    path=os.path.join(tile_dir, "generation.png"),
    description="IMAGE_INDEX is a generated image that fills in the missing parts of the masked image.",
  )

  class GenerationCheck(BaseModel):
    """A Pydantic schema for the generation check response."""

    description: str
    status: Literal["GOOD", "BAD"]
    issues: List[int]
    issues_description: Optional[str] = None

  generation_prompt = f"""
You are an advanced image analysis agent tasked with checking the output of a generative AI pipeline.

Image Descriptions:
{images.get_descriptions()}

The generative AI pipeline was given a masked image ({images.get_index("template")}) and asked to generate the missing parts of the image, resulting in the generated image ({images.get_index("generation")}).)
Your task is to triple check the generated image ({images.get_index("generation")}) to ensure that the following criteria are met:

1. The generated image must seamlessly integrate with the existing parts of the masked image ({images.get_index("template")}). There must be no gaps or inconsistencies between the existing parts of the template and the generated parts of the generated image.
2. The style of the generated image must exactly match the style of the existing parts of the masked image. This includes color palette, lighting, perspective, and overall artistic style.
3. All buildings and structures in the generated image must be complete and coherent. There should be no half-formed buildings or structures that do not make sense within the context of the scene.
4. The generated image contents and buildings must match the 3D building data as represented in the rendered view ({images.get_index("render")}). Any buildings or structures present in the rendered view must be accurately represented in the generated image.

Please provide a detailed analysis of the generated image ({images.get_index("generation")}), highlighting any areas that do not meet the above criteria. If the generated image meets all criteria, please confirm that it is acceptable, using the following output schema:
""".strip()

  contents = images.contents + [generation_prompt]

  response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents=contents,
    config={
      "response_mime_type": "application/json",
      "response_json_schema": GenerationCheck.model_json_schema(),
    },
  )

  output = GenerationCheck.model_validate_json(response.text)
  print("🔥", json.dumps(output.model_dump(), indent=2))
  return output


def main():
  config_path = os.path.join(os.path.dirname(__file__), "config.json")
  with open(config_path, "r") as f:
    config = json.load(f)
  tile_dir = config["tile_dir"]
  check_generation(tile_dir)


if __name__ == "__main__":
  main()
