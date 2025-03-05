"""
Compatibility rules for the Wave Function Collapse algorithm.
"""

# Define compatibility rules between tiles
# Directions: 0=north, 1=east, 2=south, 3=west
COMPATIBILITY_RULES = {
    "water": {
        0: ["water", "water_rock", "water_sand"],
        1: ["water", "water_rock", "water_sand"],
        2: ["water", "water_rock", "water_sand"],
        3: ["water", "water_rock", "water_sand"]
    },
    "rock": {
        0: ["rock", "water_rock", "rock_grass"],
        1: ["rock", "water_rock", "rock_grass"],
        2: ["rock", "water_rock", "rock_grass"],
        3: ["rock", "water_rock", "rock_grass"]
    },
    "sand": {
        0: ["sand", "water_sand", "sand_grass"],
        1: ["sand", "water_sand", "sand_grass"],
        2: ["sand", "water_sand", "sand_grass"],
        3: ["sand", "water_sand", "sand_grass"]
    },
    "grass": {
        0: ["grass", "sand_grass", "rock_grass"],
        1: ["grass", "sand_grass", "rock_grass"],
        2: ["grass", "sand_grass", "rock_grass"],
        3: ["grass", "sand_grass", "rock_grass"]
    },
    "water_rock": {
        0: ["water", "rock", "water_rock"],
        1: ["rock", "water_rock", "rock_grass"],
        2: ["water", "rock", "water_rock"],
        3: ["water", "water_rock", "water_sand"]
    },
    "water_sand": {
        0: ["water", "sand", "water_sand"],
        1: ["sand", "water_sand", "sand_grass"],
        2: ["water", "sand", "water_sand"],
        3: ["water", "water_rock", "water_sand"]
    },
    "sand_grass": {
        0: ["sand", "grass", "sand_grass"],
        1: ["grass", "sand_grass", "rock_grass"],
        2: ["sand", "grass", "sand_grass"],
        3: ["sand", "water_sand", "sand_grass"]
    },
    "rock_grass": {
        0: ["rock", "grass", "rock_grass"],
        1: ["grass", "sand_grass", "rock_grass"],
        2: ["rock", "grass", "rock_grass"],
        3: ["rock", "water_rock", "rock_grass"]
    }
}

def build_compatibility_matrix(tile_types, tile_to_idx):
    """
    Build a NumPy compatibility matrix from the rule definitions.
    
    Args:
        tile_types (list): List of tile type names
        tile_to_idx (dict): Mapping from tile type names to indices
        
    Returns:
        numpy.ndarray: Compatibility matrix of shape (num_tile_types, 4, num_tile_types)
    """
    import numpy as np
    
    num_tile_types = len(tile_types)
    compatibility_matrix = np.zeros((num_tile_types, 4, num_tile_types), dtype=bool)
    
    for tile, directions in COMPATIBILITY_RULES.items():
        tile_idx = tile_to_idx[tile]
        for direction, compatible_tiles in directions.items():
            for comp_tile in compatible_tiles:
                comp_tile_idx = tile_to_idx[comp_tile]
                compatibility_matrix[tile_idx, direction, comp_tile_idx] = True
                
    return compatibility_matrix