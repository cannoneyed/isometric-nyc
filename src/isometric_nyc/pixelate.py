import argparse

from PIL import Image


def pixelate_and_quantize(
  input_path: str,
  output_path: str,
  pixel_scale: int = 4,
  num_colors: int = 32,
  dither: bool = True,
) -> None:
  """
  input_path: Path to the source image.
  output_path: Where to save the result.
  pixel_scale: How 'chunky' the pixels should be. Higher = blockier.
  num_colors: The palette size (e.g., 16, 32, 64). Lower = more retro.
  dither: Whether to apply dithering (checkering) to smooth gradients.
  """

  try:
    # Load the image
    img = Image.open(input_path)

    # Calculate the new small dimensions
    # We use floor division (//) to ensure integers
    original_width, original_height = img.size
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale

    # --- STEP 1: Downscale (Create the Grid) ---
    # The key here is resample=Image.NEAREST to avoid blurring
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)

    # --- STEP 2: Color Quantization (Reduce Palette) ---
    # We use the 'quantize' method.
    # method=1 uses 'Median Cut' which is generally good for this.
    # dither=1 enables Floyd-Steinberg dithering.
    img_quantized = img_small.quantize(
      colors=num_colors, method=1, dither=1 if dither else 0
    )

    # Convert back to RGB so we can work with it normally
    img_quantized = img_quantized.convert("RGB")

    # --- STEP 3: Upscale (For Viewing) ---
    # Resize back to original size (or larger) to see the sharp pixels
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )

    # Save the result
    final_image.save(output_path)
    print(f"Success! Saved to {output_path}")
    print(f"Settings used: Scale={pixel_scale}x, Colors={num_colors}, Dither={dither}")

  except Exception as e:
    print(f"An error occurred: {e}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Pixelate and color-quantize an image for a retro pixel art look."
  )
  parser.add_argument("input", help="Path to the source image")
  parser.add_argument("output", help="Path for the output image")
  parser.add_argument(
    "-s",
    "--scale",
    type=int,
    default=4,
    help="Pixel scale factor. Higher = blockier (default: 4)",
  )
  parser.add_argument(
    "-c",
    "--colors",
    type=int,
    default=32,
    help="Number of colors in the palette (default: 32)",
  )
  parser.add_argument(
    "--no-dither", action="store_true", help="Disable dithering for a cleaner look"
  )

  args = parser.parse_args()

  pixelate_and_quantize(
    input_path=args.input,
    output_path=args.output,
    pixel_scale=args.scale,
    num_colors=args.colors,
    dither=not args.no_dither,
  )


if __name__ == "__main__":
  main()
