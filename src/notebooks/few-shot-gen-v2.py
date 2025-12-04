import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from PIL import Image
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

        output_tile_number = '000'

        contents = []

        input_image_path = os.path.join(tile_dir, output_tile_number, 'infill.png')
        input_image_ref = client.files.upload(file=input_image_path)
        contents.append(input_image_ref)

        generation_prompt = """
    You are an advanced image generation agent tasked with using the information in the provided images to generate an isometric pixel art image of a tile of New York City, in the style of classic city builder games such as SimCity 3000.

    Generation Instructions:
    The left half of the image contains pre-generated pixel art, and the right side is a 3D render of the buildings using the Google Maps 3D Tiles API. Your task is to re-generate the right side of the tile in the style of the left side (pixel art). DO NOT MODIFY THE ORIGINAL / LEFT SIDE, AND MAKE SURE ALL BUILDINGS ON THE RIGHT SIDE CORRESPOND TO THOSE IN THE SOURCE IMAGE. Use the architectural details, building shapes, and layout from the right side to strictly guide your pixel art generation. The final image should be a seamless generation, maintaining the isometric pixel art style throughout - there must be no visible "seams" between the left and right side of the image and all building textures should be consistent.

    Style Instructions:
    (((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

    Avoid: noise, texture, dithering, gradient, blur, bokeh, realistic, dirty, painting, oil painting, brushstrokes, ambient occlusion.

    Examples:
    """.strip()
        contents.append(generation_prompt)

        # Now, add the few-shot examples
        FEW_SHOT_EXAMPLE_DIRS = ['001', '002', '005', '008', '010', '012']
        for example_dir in FEW_SHOT_EXAMPLE_DIRS:
            contents.append("Example input:")
            example_image_path = os.path.join(tile_dir, example_dir, 'infill.png')
            example_image_ref = client.files.upload(file=example_image_path)
            contents.append(example_image_ref)

            contents.append("Example output:")
            example_output_image_path = os.path.join(tile_dir, example_dir, 'generation.png')
            example_output_image_ref = client.files.upload(file=example_output_image_path)
            contents.append(example_output_image_ref)

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

        output_path = os.path.join(tile_dir, output_tile_number,'few_shot_infill.png')
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif image:= part.as_image():
                print('Saving image...')
                image.save(output_path)

        output_image = Image.open(output_path)
        template_image = Image.open(input_image_path)

        return output_image, template_image

    output_image, template_image = generate_template()

    _output_image = output_image.resize((512, 512))
    _template_image = template_image.resize((512, 512))

    mo.hstack([
        mo.vstack([mo.md("**Nano Banana Image**"), _output_image]),
        mo.vstack([mo.md("**Template**"), _template_image]),
    ])
    return


@app.cell
def _(e):
    e
    return


if __name__ == "__main__":
    app.run()
