import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import dataclasses
    import json
    import os

    from pydantic import BaseModel
    from typing import List, Literal, Optional

    import marimo as mo
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from PIL import Image
    return (
        BaseModel,
        Image,
        List,
        Literal,
        Optional,
        dataclasses,
        genai,
        json,
        load_dotenv,
        mo,
        os,
        types,
    )


@app.cell
def _(load_dotenv, os):
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    return (gemini_api_key,)


@app.cell
def _(gemini_api_key, genai):
    client = genai.Client(
        api_key=gemini_api_key
    )
    return (client,)


@app.cell
def _(json, os):
    tile_dir = "/Users/andycoenen/cannoneyed/isometric-nyc/tile_plans/v01/000_001"
    references_dir = os.path.join(os.getcwd() , "references")

    view_json_path = os.path.join(tile_dir, 'view.json')
    with open(view_json_path, 'r') as f:
        view_json = json.load(f)

    latitude = view_json["lat"]
    longitude = view_json["lon"]
    return (tile_dir,)


@app.cell
def _(client, dataclasses):
    class ImageRef:  
        def __init__(self, *, id: str, index: int, path: str, description: str):
            self.id = id
            self.path = path
            self.ref = client.files.upload(file=self.path)
            self.index = f'Image {chr(ord('A') + index - 1)}'
            self.description = description.replace('IMAGE_INDEX', self.index)

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
def _(Image, Images, client, mo, os, tile_dir, types):
    def generate_template():
        aspect_ratio = "1:1" # "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
        resolution = "1K" # "1K", "2K", "4K"

        images = Images()

        images.add_image_contents(
            id='template',
            path=os.path.join(tile_dir, "template.png"),
            description="IMAGE_INDEX is a masked image that contains parts of neighboring tiles that have already been generated, with a portion of the image masked out in white."
        )

        images.add_image_contents(
            id='render',
            path=os.path.join(tile_dir, "render.png"),
            description="IMAGE_INDEX is a rendered view of the 3D building data using Google 3D tiles API"
        )

        generation_prompt = f"""
    Generate the missing half of an isometric pixel art image of a small section of New York City, in the style of classic computer city builder games such as SimCity 3000.

    Image Descriptions:
    {images.get_descriptions()}

    Style Instructions:
    (((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

    Generation Instructions:
    {images.get_index('template')} is has the right half covered in white. Please replace the white part of the masked image with the missing isometric pixel art generation. The buildings must flow seamlessly across the scene. The missing half of the masked image has pixel art buildings that correspond the 3D building data in ({images.get_index('render')}), however the style of the generated image must **exactly** match the style of the existing parts of the masked image. There must be no gaps or inconsistencies between the existing parts of the template and the generated parts of the generated image.

    DO NOT CHANGE THE EXISTING PARTS OF {images.get_index('template')}!!!
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
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            )
        )

        output_path = os.path.join(tile_dir, 'tmp.png')
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif image:= part.as_image():
                print('Saving image...')
                image.save(output_path)

        output_image = Image.open(output_path)
        template_image = Image.open(images.get_path('template'))

        return output_image, template_image

    generate_template()

    _output_image = Image.open(os.path.join(tile_dir, 'tmp.png'))
    _template_image = Image.open(os.path.join(tile_dir, 'template.png'))
    _render_image = Image.open(os.path.join(tile_dir, 'render.png'))

    mo.hstack([
        mo.vstack([mo.md("**Nano Banana Image**"), _output_image]),
        mo.vstack([mo.md("**Template**"), _template_image]),
        mo.vstack([mo.md("**Render**"), _render_image]),
    ])
    return


@app.cell
def _(BaseModel, Images, List, Literal, Optional, client, json, os, tile_dir):
    def check_generation():
        images = Images()

        images.add_image_contents(
            id='template',
            path=os.path.join(tile_dir, "template.png"),
            description="IMAGE_INDEX is a masked image that contains parts of neighboring tiles that have already been generated, with a portion of the image masked out in white."
        )

        images.add_image_contents(
            id='render',
            path=os.path.join(tile_dir, "render.png"),
            description="IMAGE_INDEX is a rendered view of the 3D building data using Google 3D tiles API"
        )

        images.add_image_contents(
            id='generation',
            path=os.path.join(tile_dir, "tmp.png"),
            description="IMAGE_INDEX is a generated image that fills in the missing parts of the masked image."
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

    The generative AI pipeline was given a masked image ({images.get_index('template')}) and asked to generate the missing parts of the image, resulting in the generated image ({images.get_index('generation')}).)
    Your task is to triple check the generated image ({images.get_index('generation')}) to ensure that the following criteria are met:

    1. The generated image must seamlessly integrate with the existing parts of the masked image ({images.get_index('template')}). There must be no gaps or inconsistencies between the existing parts of the template and the generated parts of the generated image.
    2. The style of the generated image must exactly match the style of the existing parts of the masked image. This includes color palette, lighting, perspective, and overall artistic style.
    3. All buildings and structures in the generated image must be complete and coherent. There should be no half-formed buildings or structures that do not make sense within the context of the scene.
    4. The generated image contents and buildings must match the 3D building data as represented in the rendered view ({images.get_index('render')}). Any buildings or structures present in the rendered view must be accurately represented in the generated image.

    Please provide a detailed analysis of the generated image ({images.get_index('generation')}), highlighting any areas that do not meet the above criteria. If the generated image meets all criteria, please confirm that it is acceptable, using the following output schema:
    """.strip()

        contents = images.contents + [generation_prompt]

        response = client.models.generate_content(
            model="gemini-2.5-pro",
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
