from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from perlin import generate_fractal_noise_2d


class WorldGenerator:
    """A class to generate and plot a Perlin noise-based world map."""

    def __init__(self, size=64, octaves=6, persistence=0.5):
        self.size = size
        self.octaves = octaves
        self.persistence = persistence
        self.grid = np.ones((size, size), dtype=np.int32)
        self.colors = np.array(
            [
                [156, 212, 226],  # Water
                [138, 181, 73],  # Grass
                [95, 126, 48],  # Trees
                [186, 140, 93],  # Dirt
                [220, 220, 220],  # Snow
                [245, 222, 179],  # Sand
                [140, 140, 140],  # Rocks
            ],
            dtype=np.uint8,
        )

    def generate_terrain(
        self,
        seed: int,
        water_threshold: float,
        snow_threshold: float,
        rock_probability: float,
        dirt_probability: float,
    ):
        """Generates the terrain for the world based on Perlin noise."""
        if seed is not None:
            np.random.seed(seed)

        # Generate and normalize fractal noise
        noise = generate_fractal_noise_2d(
            (self.size, self.size), (1, 1), self.octaves, self.persistence
        )
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        self.grid = np.ones((self.size, self.size), dtype=np.int32)

        # Water
        self.grid[noise < water_threshold] = 0

        # Snow
        self.grid[noise > snow_threshold] = 4

        # Sand (Beaches)
        water_mask = self.grid == 0
        sand_mask = np.zeros_like(self.grid, dtype=bool)
        ## Check for water neighbors
        sand_mask[1:-1, 1:-1] |= water_mask[1:-1, 2:]  # East
        sand_mask[1:-1, 1:-1] |= water_mask[1:-1, :-2]  # West
        sand_mask[1:-1, 1:-1] |= water_mask[2:, 1:-1]  # South
        sand_mask[1:-1, 1:-1] |= water_mask[:-2, 1:-1]  # North
        self.grid[(self.grid == 1) & sand_mask] = 5

        # Trees
        potential = ((noise - water_threshold) / (1 - water_threshold)) ** 4 * 0.7
        mask = (self.grid == 1) * (np.random.rand(self.size, self.size) < potential)
        self.grid[mask] = 2

        # Rocks
        rock_mask = (self.grid == 1) * (
            np.random.rand(self.size, self.size) < rock_probability
        )
        self.grid[rock_mask] = 6

        # Dirt
        dirt_mask = (self.grid == 1) * (
            np.random.rand(self.size, self.size) < dirt_probability
        )
        self.grid[dirt_mask] = 3

    def get_image_data(self):
        """Returns the world grid as a colored image array."""
        return self.colors[self.grid.reshape(-1)].reshape(self.grid.shape + (3,))

    def plot(self):
        """Plots the generated world grid using matplotlib."""
        image = self.get_image_data()
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    world = WorldGenerator(size=256)
    world.generate_terrain(
        seed=0,
        water_threshold=0.3,
        snow_threshold=0.8,
        rock_probability=0.02,
        dirt_probability=0.05,
    )
    world.plot()
