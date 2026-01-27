"""
Pixel art postprocessing using the unfake library.

This module wraps the unfake library to provide pixel art cleanup for AI-generated
tiles, including:
- Grid snapping (align to pixel boundaries)
- Dominant downscaling (pick most frequent color per block)
- Morphological cleanup (remove single-pixel noise)
- Jaggy cleanup (remove isolated diagonal pixels)
- Alpha binarization (convert alpha to fully opaque/transparent)
- Color quantization (map to fixed palette)

The key difference from standard unfake usage is support for a precomputed
fixed palette that's shared across all tiles in the city.
"""

# unfake library for pixel art processing
import logging
import unfake
from PIL import Image

# Suppress verbose unfake logging
logging.getLogger("unfake").setLevel(logging.WARNING)
logging.getLogger("unfake.py").setLevel(logging.WARNING)


def process_tile_unfake(
    img: Image.Image,
    pixel_size: int = 2,
    max_colors: int = 128,
    dominant_threshold: float = 0.15,
    morph_cleanup: bool = True,
    jaggy_cleanup: bool = True,
    alpha_threshold: int = 128,
    fixed_palette: list[tuple[int, int, int]] | None = None,
    snap_grid: bool = True,
    preserve_size: bool = True,
) -> Image.Image:
    """
    Apply unfake-style pixel art postprocessing to a tile.

    Args:
        img: Input PIL Image
        pixel_size: Native pixel size (default: 2)
        max_colors: Maximum colors in palette (default: 128)
        dominant_threshold: Threshold for dominant color selection (default: 0.15)
        morph_cleanup: Enable morphological noise removal (default: True)
        jaggy_cleanup: Enable jaggy edge cleanup (default: True)
        alpha_threshold: Alpha binarization threshold (default: 128)
        fixed_palette: Optional precomputed palette as list of RGB tuples
        snap_grid: Align to pixel grid (default: True)
        preserve_size: Upscale result back to original size (default: True)

    Returns:
        Processed PIL Image (same size as input if preserve_size=True)
    """
    original_size = img.size

    # Convert palette to hex format if provided
    hex_palette = None
    if fixed_palette:
        hex_palette = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in fixed_palette]

    # When using fixed_palette, set max_colors to palette size to avoid auto-detection
    effective_max_colors = len(fixed_palette) if fixed_palette else max_colors

    # Process with unfake - it accepts PIL Images directly
    result = unfake.process_image_sync(
        img,
        max_colors=effective_max_colors,
        manual_scale=pixel_size,  # Use manual_scale to skip detection
        detect_method="auto",  # Will be skipped due to manual_scale
        downscale_method="dominant",
        dom_mean_threshold=dominant_threshold,
        cleanup={"morph": morph_cleanup, "jaggy": jaggy_cleanup},
        snap_grid=snap_grid,
        alpha_threshold=alpha_threshold,
        fixed_palette=hex_palette,
        auto_color_detect=False,
    )

    output_img = result["image"]

    # Upscale back to original size if needed
    if preserve_size and output_img.size != original_size:
        output_img = output_img.resize(original_size, resample=Image.NEAREST)

    return output_img


def extract_palette_from_image(palette_img: Image.Image) -> list[tuple[int, int, int]]:
    """
    Extract RGB palette from a quantized palette image.

    Args:
        palette_img: A quantized PIL Image with a palette

    Returns:
        List of RGB tuples representing the palette colors
    """
    if palette_img.mode == "P":
        # Get palette data (flat list of R, G, B values)
        palette_data = palette_img.getpalette()
        if palette_data:
            # Convert flat list to list of RGB tuples
            colors = []
            for i in range(0, len(palette_data), 3):
                colors.append((palette_data[i], palette_data[i + 1], palette_data[i + 2]))
            # Only return colors that are actually used
            used_colors = set(palette_img.getdata())
            return [colors[i] for i in used_colors if i < len(colors)]
    elif palette_img.mode == "RGB":
        # If it's an RGB image, get unique colors
        colors = list(set(palette_img.getdata()))
        return colors[:256]  # Limit to 256
    return []


def build_palette_image_from_colors(
    colors: list[tuple[int, int, int]],
    num_colors: int = 128,
) -> Image.Image:
    """
    Build a palette image from a list of colors using median-cut quantization.

    This is similar to the existing build_unified_palette() but returns
    the actual color list as well.

    Args:
        colors: List of RGB tuples sampled from the dataset
        num_colors: Target number of colors in the palette

    Returns:
        Quantized palette image
    """
    if not colors:
        # Return a grayscale palette as fallback
        gray_colors = [(i * 2, i * 2, i * 2) for i in range(num_colors)]
        composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
        pixels = composite.load()
        for i, color in enumerate(gray_colors):
            pixels[i, 0] = color
        return composite.quantize(colors=num_colors, method=1, dither=0)

    # Create a composite image from all sampled colors
    num_pixels = len(colors)
    side = int(num_pixels**0.5) + 1

    composite = Image.new("RGB", (side, side), (0, 0, 0))
    pixels = composite.load()

    for i, color in enumerate(colors):
        x = i % side
        y = i // side
        if y < side:
            pixels[x, y] = color

    # Quantize to get the palette
    palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
    return palette_img


def postprocess_image_unfake(
    img: Image.Image,
    palette_img: Image.Image | None = None,
    pixel_size: int = 2,
    dominant_threshold: float = 0.15,
    morph_cleanup: bool = True,
    jaggy_cleanup: bool = True,
    alpha_threshold: int = 128,
    max_colors: int = 128,
) -> Image.Image:
    """
    High-level postprocessing function that matches the interface expected
    by export_dzi.py.

    Args:
        img: Input PIL Image
        palette_img: Optional palette image (if None, colors are auto-detected)
        pixel_size: Native pixel size (default: 2)
        dominant_threshold: Threshold for dominant color selection (default: 0.15)
        morph_cleanup: Enable morphological noise removal (default: True)
        jaggy_cleanup: Enable jaggy edge cleanup (default: True)
        alpha_threshold: Alpha binarization threshold (default: 128)
        max_colors: Maximum colors if no palette provided (default: 128)

    Returns:
        Processed PIL Image
    """
    # Extract palette if provided
    fixed_palette = None
    if palette_img is not None:
        fixed_palette = extract_palette_from_image(palette_img)

    return process_tile_unfake(
        img=img,
        pixel_size=pixel_size,
        max_colors=max_colors,
        dominant_threshold=dominant_threshold,
        morph_cleanup=morph_cleanup,
        jaggy_cleanup=jaggy_cleanup,
        alpha_threshold=alpha_threshold,
        fixed_palette=fixed_palette,
        snap_grid=True,
    )
