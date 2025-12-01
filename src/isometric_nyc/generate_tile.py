import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

MODEL_NAME = "gemini-3-pro-preview"
IMAGE_MODEL_NAME = "gemini-3-pro-image-preview"
REFERENCE_IMAGE_NAME = "style_a.png"


def generate_tile(
  tile_dir_path: Path,
  references_dir_path: Path,
  downscale_factor: float = 2.0,
  skip_description: bool = False,
) -> None:
  """
  Generates an isometric pixel art image for the given tile directory using Gemini.
  """
  # Load environment variables
  load_dotenv()
  gemini_api_key = os.getenv("GEMINI_API_KEY")
  if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

  client = genai.Client(api_key=gemini_api_key)

  # Validate directories
  if not tile_dir_path.exists():
    raise FileNotFoundError(f"Tile directory not found: {tile_dir_path}")
  if not references_dir_path.exists():
    raise FileNotFoundError(f"References directory not found: {references_dir_path}")

  # Load view.json
  view_json_path = tile_dir_path / "view.json"
  if not view_json_path.exists():
    raise FileNotFoundError(f"view.json not found in {tile_dir_path}")

  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  latitude = view_json.get("lat")
  longitude = view_json.get("lon")

  print(f"Processing tile at {latitude}, {longitude}...")

  # Upload the full-size reference image (render.png)
  render_path = tile_dir_path / "render.png"
  if not render_path.exists():
    raise FileNotFoundError(f"render.png not found in {tile_dir_path}")

  # Handle downscaling if requested
  if downscale_factor > 1.0:
    print(f"Downscaling render.png by factor of {downscale_factor}...")
    with Image.open(render_path) as img:
      new_width = int(img.width / downscale_factor)
      new_height = int(img.height / downscale_factor)
      resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

      # Save to temp file
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        resized_img.save(tmp_file.name)
        render_path = Path(tmp_file.name)
        print(f"Saved downscaled image to {render_path}")

  # Define prompts (copied from nano-banana.py)
  pixel_art_techniques = """
    PIXEL ART TECHNIQUES (Apply Aggressively): Translate the textures and details from image_2.png into the following pixel art conventions:

    Heavy Dithering: All gradients, shadows, and complex textures (like the arena roof, asphalt, and building facades) must be rendered using visible cross-hatch or Bayer pattern dithering. There should be NO smooth color transitions.

    Indexed Color Palette: Use a strictly limited palette (e.g., 256 colors). Colors should be flat and distinct, typical of late 90s gaming hardware.

    Aliased Edges: Every object, building, and car must have a sharp, jagged, non-anti-aliased pixel outline.

    Tiled Textures: Building windows and brickwork should look like repeating grids of pixel tiles, not realistic materials.

    Sprites: The cars in the parking lot and on the streets must be rendered as tiny, distinct pixel art sprites, not blurry dots.
    """.strip()

  description_prompt = f"""
    You are an advanced image analysis agent. Your task is to generate a checklist of no more than ten features from the attached overhead isometric render of a section of New York City. These features will be used to populate a prompt for an image generation model that will transform the input image into a stylized isometric pixel art style image in the style of SimCity 3000 based on the constraints. It's critical that the model adhere to the colors and textures of the guide image, and that's what the checklist should aim to ensure.

    The following instructions will also be provided to the model for adhering to the pixel art style - you may emphasize any of these points to ensure that the model most accurately adheres to the colors, styles, and features of the reference image. If you recognize any of the buildings or features, please refer to them by name.

    The image is an overhead isometric render of the following coordinates:
    latitude: {latitude}
    longitude: {longitude}

    {pixel_art_techniques}

    Generate *ONLY* the list of features, nothing more.
    """

  checklist = ""
  if not skip_description:
    print("Uploading render.png for analysis...")
    render_ref = client.files.upload(file=render_path)

    # Generate the checklist of features
    print("Generating feature checklist...")
    checklist_response = client.models.generate_content(
      model=MODEL_NAME,
      contents=[
        render_ref,
        description_prompt,
      ],
      config=types.GenerateContentConfig(
        response_modalities=["TEXT"],
      ),
    )
    checklist = checklist_response.text
    print(f"Checklist generated:\n{checklist}")
  else:
    print("Skipping description generation...")
    # Use render_path upload for final generation if not uploaded here
    # However, checklist is used in generation_prompt below.
    # If skipped, we should probably handle it or use a default text.
    # For now, leaving it empty or generic instructions might be best?
    # The prompt template expects {checklist}.
    checklist = "Follow the style of the reference images."

  # Prepare generation prompt
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

  # Upload assets for generation
  whitebox_path = tile_dir_path / "whitebox.png"
  if not whitebox_path.exists():
    raise FileNotFoundError(f"whitebox.png not found in {tile_dir_path}")

  reference_path = references_dir_path / REFERENCE_IMAGE_NAME
  if not reference_path.exists():
    raise FileNotFoundError(
      f"{REFERENCE_IMAGE_NAME} not found in {references_dir_path}"
    )

  print("Uploading assets for generation...")
  whitebox_ref = client.files.upload(file=whitebox_path)

  # If we skipped description, we might not have uploaded render_ref yet?
  # Wait, render_ref was uploaded in the description block above.
  # If skip_description is True, render_ref is undefined.
  if skip_description:
    print("Uploading render.png for generation...")
    render_ref = client.files.upload(file=render_path)

  reference_ref = client.files.upload(file=reference_path)

  contents = [
    generation_prompt,
    whitebox_ref,
    render_ref,
    reference_ref,
  ]

  # Check for template.png
  template_path = tile_dir_path / "template.png"
  if template_path.exists():
    print("Found template.png, uploading and updating prompt...")
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)

    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image (image_3.png). It contains parts of neighboring tiles that have already been generated.
    You MUST respect the pixels in this image that are NOT white.
    You should only generate content for the white pixels in this image, ensuring seamless continuity with the existing parts.
    The non-white pixels MUST be preserved exactly as they appear in the template.
    """

  print("Generating pixel art image...")

  response = client.models.generate_content(
    model=IMAGE_MODEL_NAME,
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio="1:1",
      ),
    ),
  )

  output_path = tile_dir_path / "generation.png"
  if output_path.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = tile_dir_path / f"generation_{timestamp}.png"

  image_saved = False
  for part in response.parts:
    if part.text is not None:
      print(f"Response text: {part.text}")

    # Check if part has image (the structure might differ slightly in library versions)
    # The notebook used: elif image:= part.as_image():
    try:
      image = part.as_image()
      if image:
        print(f"Saving image to {output_path}...")
        image.save(output_path)

        # If template exists, composite the generation with the template
        if template_path.exists():
          # Save raw generation first
          raw_output_path = tile_dir_path / "raw_generation.png"
          image.save(raw_output_path)
          print(f"Saved raw generation to {raw_output_path}")

          print("Compositing generation with template...")
          try:
            # Open the saved image
            with Image.open(output_path) as img_file:
              generated_img = img_file.convert("RGBA")

            with Image.open(template_path) as tmpl:
              # Ensure size matches
              if generated_img.size != tmpl.size:
                print(
                  f"Warning: Generated image size {generated_img.size} differs from template size {tmpl.size}. Resizing generation."
                )
                generated_img = generated_img.resize(
                  tmpl.size, Image.Resampling.LANCZOS
                )

              # Convert template to RGBA
              tmpl = tmpl.convert("RGBA")

              # Make white pixels in template transparent
              datas = tmpl.getdata()
              new_data = []
              for item in datas:
                # Check for white (allowing slight variance)
                if item[0] > 250 and item[1] > 250 and item[2] > 250:
                  new_data.append((255, 255, 255, 0))  # Transparent
                else:
                  new_data.append(item)  # Keep original pixel

              tmpl.putdata(new_data)

              # Composite: Paste template over generation
              generated_img.paste(tmpl, (0, 0), tmpl)

              # Save back to output path
              generated_img.save(output_path)
          except Exception as e:
            print(f"Error during compositing: {e}")

        image_saved = True
    except Exception as e:
      print(f"Could not save image part: {e}")

  if not image_saved:
    # Sometimes the image is in a different property or needs handling
    # If no image part, maybe it failed to generate image
    print("No image generated in response.")
  else:
    print("Generation complete.")

  # Cleanup temp file if it was created
  if downscale_factor > 1.0 and str(render_path).startswith(tempfile.gettempdir()):
    try:
      os.unlink(render_path)
      print(f"Cleaned up temp file {render_path}")
    except Exception as e:
      print(f"Error cleaning up temp file: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Generate isometric pixel art for a tile."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, whitebox.png, render.png)",
  )
  parser.add_argument(
    "--references_dir",
    default="references",
    help="Directory containing reference images (simcity.jpg)",
  )
  parser.add_argument(
    "--downscale",
    type=float,
    default=2.0,
    help="Factor to downscale the render image by (e.g. 2.0 for half size)",
  )
  parser.add_argument(
    "--skip-description",
    action="store_true",
    help="Skip the description generation step",
  )

  args = parser.parse_args()

  tile_dir = Path(args.tile_dir)
  references_dir = Path(args.references_dir)

  if not references_dir.is_absolute():
    # Assume relative to current working directory
    references_dir = Path.cwd() / references_dir

  generate_tile(
    tile_dir, references_dir, args.downscale, skip_description=args.skip_description
  )


if __name__ == "__main__":
  main()
