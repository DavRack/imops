
import numpy as np
import imageio.v3 as iio
import colour

def process_image(input_path='image.png', output_path='result.png'):
    """
    Reads an image in Linear Rec. 2020, applies a function to each pixel,
    converts it to sRGB, and saves the result.
    """
    # 1. Read the 16-bit PNG image
    try:
        image_16bit = iio.imread(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # 2. Convert to floating point [0.0, 1.0]
    image_linear = image_16bit/((2**10)-1)

    # 3. Apply a placeholder function (e.g., multiply by 1.5 and clip)
    # This is where you can insert your custom pixel manipulation function.
    processed_linear = np.clip(image_linear, 0, 1)

    # 4. Define colour spaces
    cs_rec2020_linear = colour.models.RGB_COLOURSPACE_BT2020
    cs_srgb = colour.models.RGB_COLOURSPACE_sRGB

    # 5. Convert from Linear Rec. 2020 to sRGB, applying gamma correction
    image_srgb = colour.RGB_to_RGB(
        processed_linear,
        cs_rec2020_linear,
        cs_srgb,
        apply_cctf_encoding=True
    )

    # 6. Convert to 8-bit integer range [0, 255] for saving as a standard PNG
    image_8bit = (np.clip(image_srgb, 0, 1) * 255).astype(np.uint8)

    # 7. Save the resulting image
    iio.imwrite(output_path, image_8bit)
    print(f"Processed image saved to {output_path}")

if __name__ == '__main__':
    process_image()
