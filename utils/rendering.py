"""
Rendering utilities for the Wave Function Collapse algorithm.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def render_state(result, wave, colors, map_height, map_width, tile_size, step_count=None):
    """
    Render the current state of the map using NumPy for fast image creation.
    
    Args:
        result (numpy.ndarray): Current state matrix with tile indices
        wave (numpy.ndarray): Current wave function matrix
        colors (numpy.ndarray): Color array for each tile type
        map_height (int): Map height in tiles
        map_width (int): Map width in tiles
        tile_size (int): Size of each tile in pixels
        step_count (int, optional): Current step number to display on the image
        
    Returns:
        PIL.Image: Rendered image of the current state
    """
    # Create image array of the right size
    img_height = map_height * tile_size
    img_width = map_width * tile_size
    image_array = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Process cells in batches for better performance
    for y in range(map_height):
        for x in range(map_width):
            # Get the cell's position in the image
            y_start = y * tile_size
            y_end = (y + 1) * tile_size
            x_start = x * tile_size
            x_end = (x + 1) * tile_size
            
            if result[y, x] >= 0:
                # Use the color for the selected tile type
                color = colors[result[y, x]]
            else:
                # Use grayscale based on entropy
                entropy = np.sum(wave[y, x])
                intensity = int(255 * (1 - entropy / wave.shape[2]))
                color = np.array([intensity, intensity, intensity], dtype=np.uint8)
            
            # Fill the tile area with the color using broadcasting
            image_array[y_start:y_end, x_start:x_end] = color
    
    # Create PIL Image from array
    image = Image.fromarray(image_array)
    
    # Add step count text if provided
    if step_count is not None:
        draw = ImageDraw.Draw(image)
        # Try to use a default font or fall back to the default PIL font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        # Add step count text with contrasting background for visibility
        text = f"Step: {step_count}"
        if hasattr(draw, 'textsize'):
            text_width, text_height = draw.textsize(text, font=font)
        else:
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
        
        # Draw text background for better visibility
        draw.rectangle(
            [(10, 10), (20 + text_width, 20 + text_height)],
            fill=(0, 0, 0, 128)
        )
        
        # Draw text
        draw.text((15, 15), text, fill=(255, 255, 255), font=font)
    
    return image


def save_animation(images, output_file="wave_function_collapse_animation.gif", duration=0.5, resize_factor=0.5):
    """
    Create an animation GIF from a list of images.
    
    Args:
        images (list): List of PIL.Image objects
        output_file (str): Output filename
        duration (float): Duration of each frame in seconds
        resize_factor (float): Factor to resize images for the GIF
    """
    try:
        import imageio
        
        # Resize images for the GIF
        resized_images = []
        for img in images:
            # Calculate new size
            new_width = int(img.width * resize_factor)
            new_height = int(img.height * resize_factor)
            # Resize
            img = img.resize((new_width, new_height))
            resized_images.append(np.array(img))
        
        # Create GIF
        imageio.mimsave(output_file, resized_images, duration=duration)
        
        print(f"Animation GIF created: {output_file} with {len(images)} frames")
    except ImportError:
        print("The imageio library is not installed. GIF animation not created.")


def display_image(image, title="Map generated by Wave Function Collapse"):
    """
    Display an image using matplotlib.
    
    Args:
        image (PIL.Image): Image to display
        title (str): Title for the figure
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()