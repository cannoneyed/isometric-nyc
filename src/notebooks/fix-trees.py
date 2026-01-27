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
def _():
    references_dir = "/Users/andycoenen/cannoneyed/isometric-nyc/exports/trees"
    test_dir = "/Users/andycoenen/cannoneyed/isometric-nyc/exports/trees/test"
    return references_dir, test_dir


@app.cell
def _(Image, client, mo, os, references_dir, test_dir, types):
    def fix_trees(input_image_path: str):
        aspect_ratio = "1:1" # "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
        resolution = "2K" # "1K", "2K", "4K"

        contents = []

        # Add the reference images
        reference_images = os.listdir(references_dir)
        for reference_image_path in reference_images:
            if not reference_image_path.endswith('.png'):
                continue
            contents.append("Example:")
            reference_image_path = os.path.join(references_dir, reference_image_path)
            reference_image_ref = client.files.upload(file=reference_image_path)
            contents.append(reference_image_ref)

        # Add the input image
        contents.append("Image to fix trees:")
        input_image_ref = client.files.upload(file=input_image_path)
        contents.append(input_image_ref)

        generation_prompt = """
    You are an advanced image generation agent tasked with using the information in the provided images to modify isometric pixel art images to exactly match the style of the trees in the provided images.

    Generation Instructions:

    The trees must look like the trees in the reference images - sharp, crisp edges and a limited color palette. Stick exactly to the existing distribution and shape of trees, but use the exact style from the reference images.

    DO NOT MODIFY ANYTHING IN THE IMAGE EXCEPT FOR TREES.

    Avoid: noise, texture, dithering, photorealistic, satellite
    """.strip()
        contents.append(generation_prompt)

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

        output_path = os.path.join(test_dir, 'output.png')
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif image:= part.as_image():
                print('Saving image...')
                image.save(output_path)

        input_image = Image.open(input_image_path)
        output_image = Image.open(output_path)

        return output_image, input_image

    output_image, input_image = fix_trees('/Users/andycoenen/cannoneyed/isometric-nyc/exports/trees/test/export_tl_38_-54_br_39_-53.png')

    _input_image = input_image.resize((512, 512))
    _output_image = output_image.resize((512, 512))

    mo.hstack([
        mo.vstack([mo.md("**Input**"), _input_image]),
        mo.vstack([mo.md("**Output**"), _output_image]),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
