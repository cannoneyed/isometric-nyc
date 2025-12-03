import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import dataclasses
    import json
    import os
    from typing import List, Literal, Optional

    import marimo as mo
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from PIL import Image
    from pydantic import BaseModel
    return BaseModel, List, Literal, dataclasses, genai, json, load_dotenv, os


@app.cell
def _(load_dotenv, os):
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    return (gemini_api_key,)


@app.cell
def _(gemini_api_key, genai):
    client = genai.Client(api_key=gemini_api_key)
    return (client,)


@app.cell
def _(json, os):
    tile_dir = "/Users/andycoenen/cannoneyed/isometric-nyc/tile_plans/v01/000_001"
    references_dir = os.path.join(os.getcwd(), "references")

    view_json_path = os.path.join(tile_dir, "view.json")
    with open(view_json_path, "r") as f:
      view_json = json.load(f)

    latitude = view_json["lat"]
    longitude = view_json["lon"]
    return (tile_dir,)


@app.cell
def _(dataclasses):
    class ImageRef:
      def __init__(self, *, id: str, index: int, path: str, description: str):
        self.id = id
        self.path = path
        self.index = f"Image {chr(ord('A') + index - 1)}"
        self.description = description.replace("IMAGE_INDEX", self.index)

    @dataclasses.dataclass
    class Images:
      contents: list = dataclasses.field(default_factory=list)
      descriptions: list = dataclasses.field(default_factory=list)
      refs: dict = dataclasses.field(default_factory=dict)

      def add_image_contents(self, *, id: str, path: str, description: str):
        index = len(self.contents) + 1
        image_ref = ImageRef(id=id, index=index, path=path, description=description)
        self.refs[id] = image_ref
        self.contents.append(image_ref.ref)
        self.descriptions.append(image_ref.description)

      def get_index(self, id: str):
        return self.refs[id].index

      def get_path(self, id: str):
        return self.refs[id].path

      def get_descriptions(self):
        return "\n".join(self.descriptions)
    return (Images,)


@app.cell
def _(BaseModel, List, Literal):
    class GenerationCheck(BaseModel):
        """A Pydantic schema for the generation check response."""
        
        description: str
        status: Literal["PASS", "FAIL", "FIX"]
        fixes: List[str]
    return (GenerationCheck,)


@app.cell
def _(GenerationCheck, Images, client, json, os, tile_dir):
    def check_generation():
        images = Images()
    
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
            path=os.path.join(tile_dir, "generation006.png"),
            description="IMAGE_INDEX is a generated image that fills in the missing parts of the masked image.",
        )

        generation_prompt = f"""
    You are an advanced image analysis agent tasked with checking the output of a generative AI pipeline. You must provide a PASS, FAIL, or FIX rating with justification and potential fix steps.

    Image Descriptions:
    {images.get_descriptions()}

    The generative AI pipeline was given a masked image ({images.get_index("template")}) and asked to generate the missing parts of the image, resulting in the generated image ({images.get_index("generation")}).)
    Your task is to triple check the generated image ({images.get_index("generation")}) to ensure that the following criteria are met:

    PASS/FAIL CRITERIA:
    If any of the following criteria are not met it's an automatic FAIL.

    1. The generated image must seamlessly integrate with the existing parts of the masked image ({images.get_index("template")}). There must be no gaps or inconsistencies between the existing parts of the template and the generated parts of the generated image.
    2. The style of the generated image must exactly match the style of the existing parts of the masked image. This includes color palette, lighting, perspective, and overall pixel art style.
    3. All buildings and structures in the generated image must be complete and coherent. There should be no half-formed buildings or structures that do not make sense within the context of the scene. There should be no "whitebox" rendered buildings or any buildings that are not present in the render image ({images.get_index('render')})

    FIX CRITERIA:
    If any of the below issues are present, you can output a FIX rating

    1. The generated image contents and buildings should _mostly_ match the 3D building data as represented in the rendered view ({images.get_index("render")}). Any buildings or structures present in the rendered view should be accurately represented in the generated image - some flexibility is ok but major features must be adhered to. If there's a major style discrepancy, ouput a FIX string in the following format:

    <fix>Adjust the building to be more pixel-art styled</fix>

    Examples:
        """.strip()

        contents = images.contents + [generation_prompt]

        # Now, add in the few-shot examples:
        # ====
        few_shot_images = Images()

        copied_example = GenerationCheck(
            description="The generation is a copy of the render image.",
            status="FAIL",
            fixes=[]
        )
        few_shot_images.add_image_contents(
            id='copied',
            path=os.path.join(tile_dir, "few_shots", "copied.png"),
            description=copied_example.model_dump_json()
        )

        pixelated_example = GenerationCheck(
            description="Some of the buildings are pixelated / blurry",
            status="FIX",
            fixes=["Two buildings are blurry"],
        )
        few_shot_images.add_image_contents(
            id='pixelated',
            path=os.path.join(tile_dir, "few_shots", "some_blurry.png"),
            description=pixelated_example.model_dump_json()
        )

        good_example = GenerationCheck(
            description="The generation adheres to the style of the template image and the form of the reference image",
            status="PASS",
            fixes=[]
        )
        few_shot_images.add_image_contents(
            id="good",
            path=os.path.join(tile_dir, "few_shots", "good.png"),
            description=good_example.model_dump_json()
        )

        contents = contents + few_shot_images.contents
    
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

    check_generation()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
