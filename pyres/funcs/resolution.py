def calculate_resolution(kc, pixel_size):
    """
    Calculate the resolution based on the formula: resolution = (2 * pixel_size) / kc.

    Parameters:
        kc (float): The cut-off frequency.
        pixel_size (float): The size of a pixel.

    Returns:
        float: The calculated resolution.
    """
    # Calculate the resolution
    resolution = (2 * pixel_size) / kc
    resolution = (resolution * 1000)  # Convert to nano meters
    # Print the resolution
    print(f"Resolution in nano meters(nm): {resolution:.3f}")

    return resolution



