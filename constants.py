"""
Constants for the Wave Function Collapse algorithm.
"""

# Tile definitions
TILE_TYPES = [
    "water",
    "rock",
    "sand",
    "grass",
    "water_rock",  # Transition tile between water and rock
    "water_sand",  # Transition tile between water and sand
    "sand_grass",  # Transition tile between sand and grass
    "rock_grass"   # Transition tile between rock and grass
]

# RGB Colors for visualization
TILE_COLORS = [
    [0, 105, 148],     # water - Blue
    [100, 100, 100],   # rock - Gray
    [194, 178, 128],   # sand - Beige
    [60, 130, 50],     # grass - Green
    [50, 102, 124],    # water_rock - Blue-gray
    [97, 141, 138],    # water_sand - Blue-beige
    [127, 154, 89],    # sand_grass - Beige-green
    [80, 115, 75]      # rock_grass - Gray-green
]

# Directions: 0=north, 1=east, 2=south, 3=west
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3