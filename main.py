"""
Main entry point for the Wave Function Collapse terrain generator.
"""
import random
import numpy as np
import time
import argparse

from wfc import WaveFunctionCollapse
from utils.rendering import save_animation


def set_seed(seed_value=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)


def set_rock_borders(wfc):
    """Set rock borders around the map."""
    rock_idx = wfc.tile_to_idx["rock"]
    map_height = wfc.map_height
    map_width = wfc.map_width
    
    # Top and bottom borders
    wfc.result[0, :] = rock_idx
    wfc.result[map_height-1, :] = rock_idx
    wfc.wave[0, :, :] = False
    wfc.wave[map_height-1, :, :] = False
    wfc.wave[0, :, rock_idx] = True
    wfc.wave[map_height-1, :, rock_idx] = True
    
    # Left and right borders
    wfc.result[:, 0] = rock_idx
    wfc.result[:, map_width-1] = rock_idx
    wfc.wave[:, 0, :] = False
    wfc.wave[:, map_width-1, :] = False
    wfc.wave[:, 0, rock_idx] = True
    wfc.wave[:, map_width-1, rock_idx] = True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Wave Function Collapse Terrain Generator')
    parser.add_argument('--width', type=int, default=64, help='Map width in tiles')
    parser.add_argument('--height', type=int, default=64, help='Map height in tiles')
    parser.add_argument('--tile-size', type=int, default=16, help='Tile size in pixels')
    parser.add_argument('--visualize', type=bool, default=True, help='Visualize generation steps')
    parser.add_argument('--steps-interval', type=int, default=20, help='Interval between visualization steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default="wave_function_collapse_map.png", help='Output image filename')
    parser.add_argument('--animation', type=str, default="wave_function_collapse_animation.gif", help='Animation output filename')
    parser.add_argument('--show', type=bool, default=False, help='Show the final image')
    return parser.parse_args()


def main():
    """Main function."""
    total_start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Initializing Wave Function Collapse for {args.width}x{args.height} map...")
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize the generator
    wfc = WaveFunctionCollapse(args.width, args.height, args.tile_size)
    
    # Set initial constraints
    print("Setting initial constraints...")
    set_rock_borders(wfc)
    
    # Run the algorithm
    print("Generating map...")
    step_images = wfc.run(visualize_steps=args.visualize, steps_interval=args.steps_interval)
    
    # Render the final map
    print(f"Rendering final map to {args.output}...")
    wfc.render(filename=args.output, show=args.show)
    
    # Create an animation of the generation steps
    if args.visualize and step_images:
        print(f"Creating step animation to {args.animation}...")
        save_animation(step_images, output_file=args.animation, duration=0.5, resize_factor=0.5)
    
    total_time = time.time() - total_start_time
    print(f"Map generated successfully in {total_time:.2f} seconds!")


if __name__ == "__main__":
    main()