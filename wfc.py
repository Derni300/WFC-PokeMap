"""
Implementation of the Wave Function Collapse algorithm.
"""
import random
import numpy as np
import time
from PIL import Image

from constants import TILE_TYPES, TILE_COLORS
from rules.compatibility import build_compatibility_matrix
from utils.precomputation import precompute_adjacent_cells
from utils.rendering import render_state


class WaveFunctionCollapse:
    def __init__(self, map_width, map_height, tile_size):
        """
        Initialize the Wave Function Collapse algorithm with pure NumPy optimization.
        
        Args:
            map_width (int): Width of the map in number of tiles
            map_height (int): Height of the map in number of tiles
            tile_size (int): Size of a tile in pixels
        """
        self.map_width = map_width
        self.map_height = map_height
        self.tile_size = tile_size
        
        # Definition of tile types
        self.tile_types = TILE_TYPES
        self.num_tile_types = len(self.tile_types)
        
        # Create a mapping from tile type to index for faster processing
        self.tile_to_idx = {tile: idx for idx, tile in enumerate(self.tile_types)}
        self.idx_to_tile = {idx: tile for idx, tile in enumerate(self.tile_types)}
        
        # Build compatibility matrix from rules
        self.compatibility_matrix = build_compatibility_matrix(self.tile_types, self.tile_to_idx)
        
        # Initialize wave function as a 3D boolean array [y, x, tile_type]
        # True indicates the tile type is possible at that position
        self.wave = np.ones((map_height, map_width, self.num_tile_types), dtype=bool)
        
        # Initialize result matrix with -1 indicating no tile type chosen yet
        self.result = np.full((map_height, map_width), -1, dtype=np.int8)
        
        # Colors for visualization as numpy array for faster rendering
        self.colors = np.array(TILE_COLORS, dtype=np.uint8)
        
        # Precompute adjacent cells indices for faster propagation
        self.adjacent_cells, self.adjacent_mask = precompute_adjacent_cells(map_height, map_width)
    
    def find_min_entropy_cell(self):
        """
        Find the cell with the minimum non-zero entropy using NumPy operations.
        """
        # Create a mask for uncollapsed cells
        uncollapsed_mask = self.result == -1
        
        if not np.any(uncollapsed_mask):
            return None  # All cells are collapsed
            
        # Calculate entropy for all uncollapsed cells
        entropy = np.sum(self.wave, axis=2)
        
        # Set entropy of collapsed cells to infinity to exclude them
        entropy_masked = np.where(uncollapsed_mask, entropy, np.inf)
        
        # Find minimum entropy
        min_entropy = np.min(entropy_masked)
        
        if min_entropy == np.inf or min_entropy == 0:
            return None
            
        # Find coordinates of cells with minimum entropy
        min_coords = np.where((entropy_masked == min_entropy) & (entropy_masked < np.inf))
        min_coords = list(zip(min_coords[0], min_coords[1]))
        
        # If there's a single possibility, pick it immediately
        if min_entropy == 1 and min_coords:
            return (min_coords[0][0], min_coords[0][1])
            
        # Choose a random cell among those with minimum entropy
        if min_coords:
            idx = random.randint(0, len(min_coords) - 1)
            return (min_coords[idx][0], min_coords[idx][1])
        return None
    
    def collapse_cell(self, y, x):
        """
        Collapse a cell by choosing a state from the remaining possibilities.
        """
        # Get indices of possible tile types
        possible_states = np.where(self.wave[y, x])[0]
        
        if len(possible_states) == 0:
            # If no state is possible, choose randomly from all tile types
            chosen_idx = random.randint(0, self.num_tile_types - 1)
        else:
            # Choose randomly from possible states
            chosen_idx = np.random.choice(possible_states)
        
        # Collapse the cell to the chosen state
        self.result[y, x] = chosen_idx
        
        # Update wave function - only the chosen state is possible now
        self.wave[y, x, :] = False
        self.wave[y, x, chosen_idx] = True
    
    def propagate(self, start_y, start_x):
        """
        Propagate constraints from cell (start_y, start_x) using only NumPy operations.
        """
        # Use NumPy arrays as stack/queue
        max_queue_size = self.map_width * self.map_height * 4  # Maximum possible size
        queue = np.zeros((max_queue_size, 2), dtype=np.int16)
        queue[0] = [start_y, start_x]
        queue_start = 0
        queue_end = 1
        
        # Track processed cells with a 2D boolean array
        processed = np.zeros((self.map_height, self.map_width), dtype=bool)
        
        while queue_start < queue_end:
            # Get the next cell from the queue (FIFO)
            cy, cx = queue[queue_start]
            queue_start += 1
            
            # Skip if already processed or not collapsed
            if processed[cy, cx] or self.result[cy, cx] < 0:
                continue
                
            processed[cy, cx] = True
            
            # Get the tile type index of the current cell
            current_tile_idx = self.result[cy, cx]
            
            # For each valid direction
            valid_directions = np.where(self.adjacent_mask[cy, cx])[0]
            for d in valid_directions:
                # Get adjacent cell coordinates
                ay, ax = self.adjacent_cells[cy, cx, d]
                
                # Skip if already collapsed
                if self.result[ay, ax] >= 0:
                    continue
                
                # Get the compatibility mask for this tile type and direction
                compatibility = self.compatibility_matrix[current_tile_idx, d]
                
                # Calculate the old possibilities
                old_possible = np.sum(self.wave[ay, ax])
                
                # Update the wave function using the compatibility matrix
                # Keep only possibilities that are compatible with the current cell
                self.wave[ay, ax] &= compatibility
                
                # Calculate the new possibilities
                new_possible = np.sum(self.wave[ay, ax])
                
                # If possibilities have changed, add to queue
                if new_possible < old_possible:
                    # If only one possibility left, collapse immediately
                    if new_possible == 1:
                        self.result[ay, ax] = np.where(self.wave[ay, ax])[0][0]
                    
                    # Add to queue only if not already in it
                    if queue_end < max_queue_size:
                        queue[queue_end] = [ay, ax]
                        queue_end += 1
                
                # If no possibilities left, reset to all possibilities (avoid deadlock)
                if new_possible == 0:
                    self.wave[ay, ax, :] = True
    
    def run(self, visualize_steps=False, steps_interval=100):
        """
        Run the Wave Function Collapse algorithm using NumPy for speed.
        
        Args:
            visualize_steps (bool): If True, visualize intermediate steps of generation
            steps_interval (int): Number of steps between each visualization
        
        Returns:
            list: List of PIL.Image objects representing steps if visualize_steps is True
        """
        start_time = time.time()
        step_count = 0
        images = []
        
        # Always include the initial state (step 0)
        if visualize_steps:
            images.append(self.render_current_state(step_count=0))
        
        while True:
            # Find cell with minimum entropy
            min_cell = self.find_min_entropy_cell()
            
            if min_cell is None:
                break
                
            y, x = min_cell
            
            # Collapse the cell
            self.collapse_cell(y, x)
            
            # Propagate constraints
            self.propagate(y, x)
            
            # Visualize if needed
            step_count += 1
            if visualize_steps and step_count % steps_interval == 0:
                # Capture every Nth step without time limiting
                images.append(self.render_current_state(step_count=step_count))
        
        # Always include the final state
        if visualize_steps and step_count > 0:
            images.append(self.render_current_state(step_count=step_count))
            
        elapsed = time.time() - start_time
        print(f"Generation completed in {step_count} steps and {elapsed:.2f} seconds")
        return images
    
    def render_current_state(self, step_count=None):
        """
        Render the current state of the map.
        
        Args:
            step_count (int, optional): Current step number to display on the image
        
        Returns:
            PIL.Image: Rendered image
        """
        return render_state(
            self.result, 
            self.wave, 
            self.colors, 
            self.map_height, 
            self.map_width, 
            self.tile_size, 
            step_count
        )
        
    def render(self, filename="map.png", show=False):
        """
        Render the generated map as an image.
        
        Args:
            filename (str): Filename to save the image
            show (bool): If True, display the image
        
        Returns:
            PIL.Image: Rendered image
        """
        # Create an image of the final state
        image = self.render_current_state()
        
        # Save the image
        image.save(filename)
        
        # Display the image if requested
        if show:
            from utils.rendering import display_image
            display_image(image)
        
        return image