import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os

    import marimo as mo
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from PIL import Image
    return Image, genai, json, load_dotenv, mo, os, types


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
    tile_dir = "/Users/andycoenen/cannoneyed/isometric-nyc/tile_plans/test/001"
    references_dir = os.path.join(os.getcwd() , "references")

    view_json_path = os.path.join(tile_dir, 'view.json')
    with open(view_json_path, 'r') as f:
        view_json = json.load(f)

    latitude = view_json["lat"]
    longitude = view_json["lon"]
    return latitude, longitude, references_dir, tile_dir


@app.cell
def _(client, latitude, longitude, os, tile_dir, types):
    # First, we want to use Gemini to generate a description of the image so that we can ensure that the Nano Banana generation
    # follows the guide image correctly
    pixel_art_techniques = f"""
    PIXEL ART TECHNIQUES (Apply Aggressively): Translate the textures and details from image_2.png into the following pixel art conventions:

    Heavy Dithering: All gradients, shadows, and complex textures (like the arena roof, asphalt, and building facades) must be rendered using visible cross-hatch or Bayer pattern dithering. There should be NO smooth color transitions.

    Indexed Color Palette: Use a strictly limited palette (e.g., 256 colors). Colors should be flat and distinct, typical of late 90s gaming hardware.

    Aliased Edges: Every object, building, and car must have a sharp, jagged, non-anti-aliased pixel outline.

    Tiled Textures: Building windows and brickwork should look like repeating grids of pixel tiles, not realistic materials.

    Sprites: The cars in the parking lot and on the streets must be rendered as tiny, distinct pixel art sprites, not blurry dots.
    """.strip()

    description_prompt = f"""
    You are an advanced image analysis agent. Your task is to generate a checklist of no more than five features from the attached overhead isometric render of a section of New York City. These features will be used to populate a prompt for an image generation model that will transform the input image into a stylized isometric pixel art style image in the style of SimCity 3000 based on the constraints. It's critical that the model adhere to the colors and textures of the guide image, and that's what the checklist should aim to ensure.

    The following instructions will also be provided to the model for adhering to the pixel art style - you may emphasize any of these points to ensure that the model most accurately adheres to the colors, styles, and features of the reference image.

    The image is an overhead isometric render of the following coordinates:
    latitude: {latitude}
    longitude: {longitude}

    {pixel_art_techniques}

    Generate *ONLY* the list of features, nothing more.
    """

    # Upload the full-size reference image
    _render_path = os.path.join(tile_dir, "render.png")
    _render_ref = client.files.upload(file=_render_path)

    # Generate the checklist of features
    _response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[
            _render_ref,
            description_prompt,
        ],
        config=types.GenerateContentConfig(
            response_modalities=['TEXT'],
        )
    )

    # Print the checklist
    checklist = _response.text
    print(checklist)
    return checklist, pixel_art_techniques


@app.cell
def _(checklist, pixel_art_techniques):
    generation_prompt = f"""
    Generate a low-resolution, isometric pixel art conversion of the provided reference image, strictly adhering to the visual style of late 1990s PC strategy games like SimCity 3000.

    It should look like a 640x480 game screenshot stretched onto a modern monitor.

    Do not generate high-definition "voxel art" or smooth digital paintings. The aesthetic must be crunchy, retro, and low-fi. Ensure you stick to a low-fi SVGA color palette.

    image_0.png is the whitebox geometry - the final image must adhere to the shapes of the buildings defined here.

    image_1.png is the 3D render of the city - use this image as a reference for the details, textures, colors, and lighting of the buildings, but DO NOT JUST downsample the pixels - we want to use the style of image 3.

    image_3.jpg is a reference image for the style of SimCity 3000 pixel art - you MUST use this style for the pixel art generation.

    CRITICAL: STYLE OVER REALISM

    Do NOT simply downsample or blur the photorealistic reference image (image_2.png).

    The result must NOT look like a low-resolution photograph.

    It MUST look like a piece of pixel art that was painstakingly drawn pixel-by-pixel using a limited color palette. The aesthetic should be "crunchy," "retro," and "low-fi."

    GEOMETRY & COMPOSITION (Adhere Strictly):

    Use the white masses in image_0.png as the blueprint for all building shapes and locations.

    {pixel_art_techniques}

    Instructions:
    {checklist}
    """
    return (generation_prompt,)


@app.cell
def _(
    Image,
    client,
    generation_prompt,
    mo,
    os,
    references_dir,
    tile_dir,
    types,
):
    aspect_ratio = "1:1" # "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
    resolution = "1K" # "1K", "2K", "4K"

    whitebox_path = os.path.join(tile_dir, "whitebox.png")
    render_path = os.path.join(tile_dir, "render.png")
    reference_path = os.path.join(references_dir, "simcity.jpg")

    whitebox_ref = client.files.upload(file=whitebox_path)
    render_ref = client.files.upload(file=render_path)
    reference_ref = client.files.upload(file=reference_path)

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            generation_prompt,
            whitebox_ref,
            render_ref,
            reference_ref
        ],
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        )
    )

    output_path = os.path.join(tile_dir, 'generation.png')
    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif image:= part.as_image():
            print('Saving image...')
            image.save(output_path)

    _output_image = Image.open(output_path)

    mo.hstack([
        mo.vstack([mo.md("**Nano Banana Image**"), _output_image]),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
