"""
Shared image preprocessing functions for synthetic data generation and inference.

These transformations are applied to render regions to create training data and
must be applied identically during inference to match the training distribution.

Transformations (applied in order):
1. desaturation: Reduce color saturation (0.0 to 1.0) - kills the "satellite green"
2. noise: Add multiplicative grain (0.0 to 1.0) - mimics texture loss, looks like gritty paper
3. gamma_shift: Apply gamma crush (0.0 to 1.0) - THE SECRET SAUCE
   * Pushes dark greys to black while keeping lighter areas visible
   * Separates "tree tops" (visible) from "ground" (black)
   * Destroys the flat look of satellite photos
   * At intensity=1.0: gamma=1.8 + 0.7x brightness (the perfect corruption)
"""

import numpy as np
from PIL import Image


def apply_noise(img: Image.Image, intensity: float) -> Image.Image:
  """
  Add multiplicative noise (grain) to an image.

  Uses multiplicative noise to mimic "texture loss" rather than "sensor static."
  This looks like gritty paper or rough ground, encouraging the model to generate
  textured pixel art.

  Args:
    img: Input PIL Image
    intensity: Noise intensity (0.0 = no grain, 1.0 = maximum graininess)

  Returns:
    Image with grain applied
  """
  if intensity <= 0:
    return img

  # Convert to numpy array
  img_array = np.array(img, dtype=np.float32)

  # Generate multiplicative noise centered at 1.0
  # At intensity=1.0, scale = 0.15 (15% variation)
  # This multiplies the texture rather than adding static
  noise_multiplier = np.random.normal(
    loc=1.0, scale=intensity * 0.15, size=img_array.shape
  )

  # Apply multiplicative noise
  noisy_array = img_array * noise_multiplier

  # Clip to valid range
  noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)

  return Image.fromarray(noisy_array)


def apply_desaturation(img: Image.Image, intensity: float) -> Image.Image:
  """
  Desaturate an image (move towards grayscale).

  Args:
    img: Input PIL Image
    intensity: Desaturation intensity (0.0 = no change, 1.0 = full grayscale)

  Returns:
    Desaturated image
  """
  if intensity <= 0:
    return img

  # Convert to numpy array
  img_array = np.array(img, dtype=np.float32)

  # Calculate grayscale using standard weights (ITU-R BT.601)
  grayscale = (
    0.299 * img_array[:, :, 0]
    + 0.587 * img_array[:, :, 1]
    + 0.114 * img_array[:, :, 2]
  )

  # Expand grayscale to 3 channels
  grayscale_rgb = np.stack([grayscale, grayscale, grayscale], axis=2)

  # Blend between original and grayscale based on intensity
  result = (1 - intensity) * img_array + intensity * grayscale_rgb

  return Image.fromarray(result.astype(np.uint8))


def apply_gamma_shift(img: Image.Image, intensity: float) -> Image.Image:
  """
  Apply gamma crush to destroy the flat look and separate highlights from shadows.

  The "Gamma Crush" pushes dark greys to black while keeping lighter areas visible.
  This separates "tree tops" (which stay visible) from "ground" (which goes black),
  forcing the model to learn depth and lighting rather than just recoloring flat images.

  Args:
    img: Input PIL Image
    intensity: Gamma crush intensity (0.0 = no change, 1.0 = full crush with gamma=1.8)

  Returns:
    Image with gamma crush applied and optional final brightness reduction
  """
  if intensity <= 0:
    return img

  # Convert to numpy array and normalize to [0, 1]
  img_array = np.array(img, dtype=np.float32) / 255.0

  # Apply gamma crush
  # intensity 0.0 -> gamma 1.0 (no change)
  # intensity 1.0 -> gamma 1.8 (the sweet spot - crushes blacks, preserves highlights)
  gamma = 1.0 + (intensity * 0.8)
  crushed = np.power(img_array, gamma)

  # Apply final brightness reduction at higher intensities
  # This simulates very poor lighting conditions
  if intensity > 0.5:
    brightness_factor = 1.0 - (intensity - 0.5) * 0.6  # Goes from 1.0 to 0.7
    crushed = crushed * brightness_factor

  # Convert back to [0, 255]
  result = (crushed * 255).astype(np.uint8)

  return Image.fromarray(result)


def apply_preprocessing(
  img: Image.Image,
  desaturation: float = 0.0,
  noise: float = 0.0,
  gamma_shift: float = 0.0,
) -> Image.Image:
  """
  Apply all preprocessing transformations in the correct order.

  Order matters! The "Perfect Corruption" recipe:
  1. Desaturate to kill the satellite green and prevent color overfitting
  2. Add grain to mimic texture loss (gritty paper look)
  3. Gamma crush to separate highlights from shadows and destroy flatness

  Args:
    img: Input PIL Image (RGB or RGBA)
    desaturation: Desaturation intensity (0.0 to 1.0)
    noise: Noise intensity (0.0 to 1.0)
    gamma_shift: Gamma shift intensity (0.0 to 1.0)

  Returns:
    Preprocessed PIL Image
  """
  # Ensure RGB mode for processing (preserve alpha if present)
  has_alpha = img.mode == "RGBA"
  if has_alpha:
    alpha = img.split()[-1]
    img_rgb = img.convert("RGB")
  else:
    img_rgb = img

  # Apply transformations in order
  result = img_rgb
  if desaturation > 0:
    result = apply_desaturation(result, desaturation)
  if noise > 0:
    result = apply_noise(result, noise)
  if gamma_shift > 0:
    result = apply_gamma_shift(result, gamma_shift)

  # Restore alpha if it was present
  if has_alpha:
    result = result.convert("RGBA")
    result.putalpha(alpha)

  return result
