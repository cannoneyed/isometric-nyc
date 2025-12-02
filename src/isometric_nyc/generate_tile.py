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

from isometric_nyc.create_template import create_template

MODEL_NAME = "gemini-2.5-flash"
IMAGE_MODEL_NAME = "gemini-3-pro-image-preview"
REFERENCE_IMAGE_NAME = "style_c.png"


def generate_tile(
  tile_dir_path: Path,
  references_dir_path: Path,
  downscale_factor: float = 4.0,
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

  # Full-size render path (used for feature checklist)
  render_path_full = tile_dir_path / "render.png"
  if not render_path_full.exists():
    raise FileNotFoundError(f"render.png not found in {tile_dir_path}")

  # Downscaled render path (used for image generation)
  render_path_downscaled: Path | None = None
  if downscale_factor > 1.0:
    print(f"Downscaling render.png by factor of {downscale_factor}...")
    with Image.open(render_path_full) as img:
      new_width = int(img.width / downscale_factor)
      new_height = int(img.height / downscale_factor)
      resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

      # Save to temp file
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        resized_img.save(tmp_file.name)
        render_path_downscaled = Path(tmp_file.name)
        print(f"Saved downscaled image to {render_path_downscaled}")

  # Define prompts (copied from nano-banana.py)
  pixel_art_techniques = """
    PIXEL ART TECHNIQUES (Apply Aggressively): Translate the textures and details from <render> into the following pixel art conventions:

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
    # Use full-size render for feature analysis (better detail)
    print("Uploading full-size render.png for analysis...")
    render_ref_full = client.files.upload(file=render_path_full)

    # Generate the checklist of features
    print("Generating feature checklist...")
    checklist_response = client.models.generate_content(
      model=MODEL_NAME,
      contents=[
        render_ref_full,
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
    checklist = "Follow the style of the reference images."

  # Prepare generation prompt
  generation_prompt = """
<render> is the 3D render of the city - use this image as a reference for the details, textures, colors, and lighting of the buildings, but DO NOT  downsample the pixels - we want to use the style of <reference>.

<reference> is a reference image for the style of SimCity 3000 pixel art - you MUST use this style for the pixel art generation.

Use the white masses in <whitebox> as the blueprint for all building shapes and locations. Check carefully to make sure every building in <whitebox> and <render> is present in the generation, and ensure that the colors and textures of the buildings are correct.

Style Instructions:
(((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.
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

  # Upload render for generation (use downscaled version if available)
  render_path_for_generation = render_path_downscaled or render_path_full
  print(
    f"Uploading render.png for generation (downscaled: {render_path_downscaled is not None})..."
  )
  render_ref = client.files.upload(file=render_path_for_generation)

  reference_ref = client.files.upload(file=reference_path)
  whitebox_prefix = "This is a whitebox geometry of isometric render of a section of New York City. We'll refer to this as <whitebox>."
  render_prefix = "This is a rendered view of the 3D building data using Google 3D tiles API. We'll refer to this as <render>."
  reference_prefix = "This is a reference image for the style of SimCity 3000 pixel art. We'll refer to this as <reference>."

  image_contents = [
    whitebox_prefix,
    whitebox_ref,
    render_prefix,
    render_ref,
    reference_prefix,
    reference_ref,
  ]

  # Create template from neighbors if they exist
  print("Checking for neighbors to create template...")
  create_template(tile_dir_path)

  # Check for template.png
  template_path = tile_dir_path / "template.png"
  if template_path.exists():
    print("Found template.png, uploading and updating prompt...")
    template_prefix = "This is a template image that contains parts of neighboring tiles that have already been generated. We'll refer to this as <template>."
    template_ref = client.files.upload(file=template_path)
    image_contents.append(template_prefix)
    image_contents.append(template_ref)

    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    generation_prompt += """
Generation Instructions:
The last image provided is a template image <template>. It contains parts of neighboring tiles that have already been generated. The white areas are empty and need to be filled.
    
You must continue the generation by filling in the white areas, using the <whitebox> geometry and <render> as guides.
    
The existing neighbor parts (the colorful pixel art sections) MUST be preserved exactly as they appear in the template.
    """

  print("Generating pixel art image...")
  prompt_contents = image_contents + [generation_prompt]

  print("🔥🔥🔥 Prompt:")
  for content in prompt_contents:
    if isinstance(content, str):
      print(content)

  response = client.models.generate_content(
    model=IMAGE_MODEL_NAME,
    contents=prompt_contents,
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
                # Pure white (255, 255, 255) is the background color from create_template.py
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
  if render_path_downscaled and str(render_path_downscaled).startswith(
    tempfile.gettempdir()
  ):
    try:
      os.unlink(render_path_downscaled)
      print(f"Cleaned up temp file {render_path_downscaled}")
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
