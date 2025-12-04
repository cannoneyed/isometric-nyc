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
    return Image, genai, load_dotenv, mo, os, types


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
def _(os):
    tile_dir = "/Users/andycoenen/cannoneyed/isometric-nyc/synthetic_data/tiles/v02"
    references_dir = os.path.join(os.getcwd() , "references")
    return (tile_dir,)


@app.cell
def _(Image, client, mo, os, tile_dir, types):
    def generate_template():
        aspect_ratio = "1:1" # "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
        resolution = "2K" # "1K", "2K", "4K"

        output_tile_number = '006'

        contents = []

        input_image_path = os.path.join(tile_dir, output_tile_number, 'composition.png')
        input_image_ref = client.files.upload(file=input_image_path)
        contents.append(input_image_ref)

        generation_prompt = f"""
    You are an advanced image generation agent tasked with using the information in the provided images to generate an isometric pixel art image of a tile of New York City, in the style of classic city builder games such as SimCity 3000.

    Generation Instructions:
    The upper left quadrant is a whitebox render of the buildings, and the upper right quadrant is a 3D render of the buildings using the Google Maps 3D Tiles API. Your task is to finish the masked pixel art version of the tile in the lower left quadrant - fill in all missing pixels to complete the image based on the templates above. ENSURE THAT THE GENERATED IMAGE EXACTLY AND ONLY FILLS THE LOWER RIGHT QUADRANT OF THE IMAGE - DO NOT MODIFY THE OTHER QUADRANTS. Use the architectural details, building shapes, and layout from the upper quadrants to inform your pixel art generation. The final image should be a seamless blend of the provided templates, maintaining the isometric pixel art style throughout.

    Style Instructions:
    (((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

    Examples:
    """.strip()
        contents.append(generation_prompt)

        # Now, add the few-shot examples
        FEW_SHOT_EXAMPLE_DIRS = ['000', '001', '002', '005']
        for example_dir in FEW_SHOT_EXAMPLE_DIRS:
            example_image_path = os.path.join(tile_dir, example_dir, 'composition.png')
            example_image_ref = client.files.upload(file=example_image_path)
            contents.append(example_image_ref)

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
                    image_size=resolution,
                ),
            )
        )

        output_path = os.path.join(tile_dir, output_tile_number,'few_shot_composition.png')
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif image:= part.as_image():
                print('Saving image...')
                image.save(output_path)

        output_image = Image.open(output_path)
        composition_image = Image.open(input_image_path)

        return output_image, composition_image

    output_image, composition_image = generate_template()

    _output_image = output_image.resize((512, 512))
    _composition_image = composition_image.resize((512, 512))

    mo.hstack([
        mo.vstack([mo.md("**Nano Banana Image**"), _output_image]),
        mo.vstack([mo.md("**Composition**"), _composition_image]),
    ])

    return


@app.cell
def _(Image, mo, os, tile_dir):
    # The output_tile_number was defined in a local scope, so we redefine it here.
    # To avoid potential conflicts, we'll use a private variable name.
    _output_tile_number = '006'

    # Construct the path to the generated image from the previous cell
    _generated_composition_path = os.path.join(
        tile_dir, _output_tile_number, "few_shot_composition.png"
    )

    # Open the image
    _generated_image = Image.open(_generated_composition_path)

    # Get the dimensions of the image
    _width, _height = _generated_image.size

    # Define the crop box for the lower-right quadrant
    # The box is a 4-tuple: (left, upper, right, lower)
    _crop_box = (_width // 2, _height // 2, _width, _height)

    # Crop the image
    _lower_right_quadrant = _generated_image.crop(_crop_box)

    # Resize the cropped quadrant to 1024x1024 using a high-quality filter
    _target_size = (1024, 1024)
    spliced_image = _lower_right_quadrant.resize(
        _target_size, Image.Resampling.LANCZOS
    )

    # Construct the output path for the new image
    _output_path = os.path.join(
        tile_dir, _output_tile_number, "few_shot_generation.png"
    )

    # Save the final image
    spliced_image.save(_output_path)

    # Display the spliced and resized image
    mo.vstack([
        mo.md(f"**Spliced and Resized Image** (`1024x1024`)"),
        spliced_image
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
