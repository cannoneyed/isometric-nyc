# Water Fix

Many quadrants in the generation DB have incorrect color water - the aim of this
new functionality is to correct this at scale.

An advanced AI agent (Gemini) gave me a target implementation of a "soft blend",
where we replace a given target color with a given final color:

Here's what they said about it:

## Gemini Response

This is a great question that moves past basic image processing into concepts
used in professional compositing.

The issue with the previous "Global Threshold" approach is that it creates a
**hard, binary mask**. A pixel is either deemed "water" and completely replaced,
or "not water" and left untouched.

In reality, edge pixels (anti-aliasing) are a mixture. A pixel on the edge of
the truss might be 60% brown and 40% water. To look natural, your replacement
color needs to respect that 40% opacity.

To achieve this, we need a **Soft Mask** (also known as an Alpha Matte). Instead
of being just black or white, this mask will contain shades of gray indicating
_how much_ a pixel resembles the target color.

Here is the implementation using NumPy to calculate color distances and perform
alpha blending.

### The "Soft Blend" Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def soft_color_replace(image, target_color, new_color, blend_softness):
    """
    Replaces a color while preserving anti-aliased edges using a soft mask based on color distance.

    Args:
        image (np.array): Input image (RGB, uint8).
        target_color (tuple): (R, G, B) color to replace (uint8).
        new_color (tuple): (R, G, B) replacement color (uint8).
        blend_softness (float): Controls the width of the blend transition.
                                Higher values = wider, softer edges. Try ranges 20.0 - 100.0.

    Returns:
        tuple: (result_image_uint8, soft_mask_uint8)
    """
    # --- PREPARATION ---
    # Convert everything to floats between 0.0 and 1.0 for accurate math.
    # Doing math on uint8 directly leads to overflow/underflow errors.
    img_float = image.astype(np.float32) / 255.0
    target_float = np.array(target_color, dtype=np.float32) / 255.0
    new_color_float = np.array(new_color, dtype=np.float32) / 255.0

    # Normalize softness parameter to 0-1 scale roughly matching color distance space
    # (Max Euclidean distance in unit RGB cube is sqrt(3) ~= 1.732)
    softness_scale = (blend_softness / 255.0) * np.sqrt(3)
    # Ensure it's not zero to avoid division errors
    softness_scale = max(softness_scale, 1e-5)


    # --- STEP 1: Calculate Color Distance Map ---
    # Find Euclidean distance between every pixel's color and the target color.
    # axis=2 calculates the norm across the R,G,B channels.
    # Result shape is (H, W). 0.0 means exact match, higher means very different.
    distances = np.linalg.norm(img_float - target_float, axis=2)


    # --- STEP 2: Create the Soft Mask (Alpha Matte) ---
    # We invert the distances: 0 distance should be 1.0 opacity.
    # We divide by softness_scale to control how quickly opacity drops off.
    alpha_mask = 1.0 - (distances / softness_scale)

    # Clip results to ensure the mask stays strictly between 0.0 and 1.0
    alpha_mask = np.clip(alpha_mask, 0.0, 1.0)

    # Reshape mask from (H, W) to (H, W, 1) so it can be multiplied with RGB images later
    alpha_expanded = alpha_mask[:, :, np.newaxis]


    # --- STEP 3: Perform Alpha Blending ---
    # Create a solid image filled entirely with the new color
    solid_new_color_img = np.full_like(img_float, new_color_float)

    # Standard Alpha Compositing Formula:
    # Final = (Foreground * Alpha) + (Background * (1 - Alpha))
    # Foreground is the new color, Background is the original image.
    blended_float = (solid_new_color_img * alpha_expanded) + \
                    (img_float * (1.0 - alpha_expanded))


    # --- FINALIZE ---
    # Convert back to uint8 format for display
    result_img = (blended_float * 255).astype(np.uint8)
    soft_mask_uint8 = (alpha_mask * 255).astype(np.uint8)

    return result_img, soft_mask_uint8

# --- DEMONSTRATION ---

img_bgr = cv2.imread('image_0.png') # Ensure this matches your filename
if img_bgr is None:
    print("Error loading image.")
    exit()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 1. Define parameters
# Sample water color (approximate from image)
water_color = (75, 105, 125)

# Bright color to make the effect obvious
replacement_col = (255, 50, 255) # Bright Magenta

# **Crucial Parameter**: Controls how wide the soft edge is.
# Too low (e.g., 5) will look like the previous hard threshold method.
# Too high (e.g., 150) will start tinting the bridge structure itself.
softness_val = 60.0

# 2. Run Soft Replacement
result_soft, mask_soft = soft_color_replace(img_rgb, water_color, replacement_col, softness_val)

# 3. Visualization
# Zooming in to see the edge quality clearly
crop_y, crop_x = 250, 450
crop_h, crop_w = 100, 100
img_crop = img_rgb[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
mask_crop = mask_soft[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
result_crop = result_soft[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))

# Top Row: Full Images
ax[0,0].imshow(img_rgb)
ax[0,0].set_title("Original")
ax[0,1].imshow(mask_soft, cmap='gray')
ax[0,1].set_title(f"Soft Mask (Softness: {softness_val})")
ax[0,2].imshow(result_soft)
ax[0,2].set_title("Soft Blend Result")

# Bottom Row: Zoomed Crops to show edge quality
ax[1,0].imshow(img_crop)
ax[1,0].set_title("Original Crop (Note Anti-aliasing)")
ax[1,1].imshow(mask_crop, cmap='gray')
ax[1,1].set_title("Soft Mask Crop (Gray edges)")
ax[1,2].imshow(result_crop)
ax[1,2].set_title("Result Crop (Preserved blend)")

for a in ax.flatten(): a.axis('off')
plt.tight_layout()
plt.show()
```

### How this works technically

1.  **Floating Point Math:** We convert the image to floats (0.0 - 1.0).
    Standard image math happens on integers (0-255), which loses the precision
    necessary for subtle blending calculations.
2.  **Color Distance (Euclidean Norm):**
    `np.linalg.norm(img_float - target_float, axis=2)` Instead of asking "Is
    this pixel in range? (Yes/No)", we ask: "Exactly how far away is this
    pixel's color from the water color in 3D RGB space?"
3.  **The Soft Mask (Alpha):** We take that distance and invert it.
    - If distance is 0 (perfect match), the mask is 1.0 (fully replace).
    - If distance is high (bridge truss), the mask becomes 0.0 (don't touch).
    - **Crucially**, if the distance is small (an edge pixel that is mostly
      water but slightly brown), the mask might be 0.8.
4.  **Alpha Blending:** The final step uses the standard compositing formula:
    $Result = (NewColor \times Mask) + (OriginalPixel \times (1.0 - Mask))$ If
    the mask is 0.8 at an edge pixel, the final pixel becomes 80% bright magenta
    and 20% original brownish-water, perfectly preserving the anti-aliasing.

## Task

Your task is to implement a new script in the `src/isometric_nyc/generation`
dir called "replace_color.py" that takes the following parameters:

- generation_dir - the directory containing the generation db
- x - the x coordinate of the quadrant to edit
- y - the y coordinate of the quadrant to edit
- target_color - a hex code of the color to be replaced
- replacement_color - a hex code of the color to replace it with

Then, implement the code to load the png image pixels and update the pixels. Add
a --dry-flag that just exports the updated png to the exports subdir of the
generation dir without saving it back to the db.
