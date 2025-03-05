"""
Precomputation utilities for the Wave Function Collapse algorithm.
"""
import numpy as np


def precompute_adjacent_cells(map_height, map_width):
    """
    Precompute adjacent cells for each position for faster lookups.
    
    Args:
        map_height (int): Height of the map in tiles
        map_width (int): Width of the map in tiles
        
    Returns:
        tuple: (adjacent_cells, adjacent_mask)
            - adjacent_cells: Array of shape (map_height, map_width, 4, 2) with adjacent cell coordinates
            - adjacent_mask: Boolean array of shape (map_height, map_width, 4) indicating valid adjacent cells
    """
    # Create arrays to store adjacent cells [y, x, direction, 2]
    # Last dimension stores (y, x) of adjacent cell
    adjacent_cells = np.full((map_height, map_width, 4, 2), -1, dtype=np.int16)
    adjacent_mask = np.zeros((map_height, map_width, 4), dtype=bool)
    
    # Precompute all valid adjacent cells
    for y in range(map_height):
        for x in range(map_width):
            # North (0)
            if y > 0:
                adjacent_cells[y, x, 0] = [y-1, x]
                adjacent_mask[y, x, 0] = True
            # East (1)
            if x < map_width - 1:
                adjacent_cells[y, x, 1] = [y, x+1]
                adjacent_mask[y, x, 1] = True
            # South (2)
            if y < map_height - 1:
                adjacent_cells[y, x, 2] = [y+1, x]
                adjacent_mask[y, x, 2] = True
            # West (3)
            if x > 0:
                adjacent_cells[y, x, 3] = [y, x-1]
                adjacent_mask[y, x, 3] = True
                
    return adjacent_cells, adjacent_mask