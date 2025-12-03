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

Your task is to grade {images.get_index("generation")}. To pass, {images.get_index("generation")} must successfully translate the **facts** of {images.get_index("render")} into the **style** of {images.get_index("template")}.

### Step 1: Visual Inventory (Mental Scratchpad)
Before providing a verdict, compare the dominant buildings in the Ground Truth Render ({images.get_index("render")}) vs the Generated Result ({images.get_index("generation")}). 
- Do the colors of the major facades match? (e.g., if the render has a black glass tower, the pixel art must be dark/black).
- Do the material textures match? (e.g., brick vs. glass vs. concrete).
- Do the relative heights match?

### Step 2: Evaluation Criteria

1. **Perfect Continuity (Pass/Fail):** The left half of {images.get_index("generation")} must EXACTLY match the unmasked parts of {images.get_index("template")}. No shifting, blurring, or color changes at the seam.

2. **Strict Style Adherence (Pass/Fail):** The generated section must be distinct, sharp Pixel Art (hard edges, specific color palette, sprites).
   - It must **NOT** look like a blurry photo or a 3D render.
   - Roads must be flat pixel surfaces, not textured asphalt.
   - Cars must be pixel sprites, not blurry blobs.

3. **Semantic Fidelity to Render:** - While the *style* must differ from {images.get_index("render")}, the *architectural identity* must match. 
   - **Color Matching:** The color palette of specific buildings in {images.get_index("generation")} must reflect the local colors in {images.get_index("render")}. If a building is dark/black in the render, it must be dark in the pixel art. If it is beige brick in the render, it must be beige pixel art.
   - **Material Translation:** A glass building in the render should look like a reflective surface in pixel art; a concrete building should look matte.
   - **Hallucination Check:** The generator must not invent new building styles that do not exist in the render just to satisfy the pixel art aesthetic.

4. **Coherence:** No half-formed buildings or Escher-like geometry errors.

5. **No Gaps:** No empty white space remaining.

### Output
If the image fails any criteria, explain specifically why (e.g., "Failure: The main skyscraper is black glass in the render but appears as white concrete in the generation").
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

  if output.status == "GOOD":
    print("The first checker is good! Prompt the user for manual feedback.")
    return True
  else:
    return False


def main():
  config_path = os.path.join(os.path.dirname(__file__), "config.json")
  with open(config_path, "r") as f:
    config = json.load(f)
  tile_dir = config["tile_dir"]
  check_generation(tile_dir)


if __name__ == "__main__":
  main()
