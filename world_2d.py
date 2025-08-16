# world_2d.py
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from perlin_numba import generate_fractal_noise_2d


class WorldGenerator:
    """A class to generate and plot a Perlin noise-based world map."""

    def __init__(self, size=256, octaves=6, persistence=0.5):
        self.size = size
        self.octaves = octaves
        self.persistence = persistence
        self.grid = np.ones((size, size), dtype=np.int32)
        self.colors = np.array(
            [
                [25, 129, 148],  # Deep Water
                [42, 169, 190],  # Shallow Water
                [110, 165, 87],  # Grass
                [85, 140, 65],  # Forest (Trees)
                [195, 155, 119],  # Dirt
                [220, 220, 220],  # Snow
                [245, 222, 179],  # Sand
                [140, 140, 140],  # Rocks
            ],
            dtype=np.uint8,
        )
        self.noise = None

    def _generate_noise(self, seed: int):
        """Generates and normalizes the fractal Perlin noise."""
        if seed is not None:
            np.random.seed(seed)

        self.noise = generate_fractal_noise_2d(
            (self.size, self.size), (1, 1), self.octaves, self.persistence
        )
        self.noise = (self.noise - self.noise.min()) / (
            self.noise.max() - self.noise.min()
        )
        self.grid = np.ones((self.size, self.size), dtype=np.int32)

    def _apply_biomes(self, water_threshold: float, snow_threshold: float):
        """Applies biomes like water, sand, snow, grass, and trees."""

        # Set a base biome for all land areas
        land_mask = self.noise >= water_threshold
        self.grid[land_mask] = 2  # Grass

        # Water (lowest elevations)
        deep_water_threshold = water_threshold * 0.7
        self.grid[self.noise < deep_water_threshold] = 0  # Deep Water
        self.grid[
            (self.noise >= deep_water_threshold) & (self.noise < water_threshold)
        ] = 1  # Shallow Water

        # Sand (Beaches)
        sand_mask = np.zeros_like(self.grid, dtype=bool)
        water_mask = (self.grid == 0) | (self.grid == 1)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                shifted_mask = np.roll(water_mask, (i, j), axis=(0, 1))
                sand_mask |= shifted_mask

        self.grid[(self.grid == 2) & sand_mask] = 6

        # Snow (highest elevations)
        if snow_threshold > 0:
            self.grid[self.noise > (1 - snow_threshold)] = 5

        # Trees (Forest)
        tree_mask = (self.grid == 2) & (self.noise > 0.35) & (self.noise < 0.65)
        self.grid[tree_mask] = 3

        # Dirt patches (lower grassy areas)
        dirt_mask = (self.grid == 2) & (self.noise < 0.45)
        self.grid[dirt_mask] = 4

    def _apply_features(self, rock_probability: float, dirt_probability: float):
        """Applies small terrain features like rocks and dirt."""
        # Rocks
        rock_mask = (self.grid == 2) * (
            np.random.rand(self.size, self.size) < rock_probability
        )
        self.grid[rock_mask] = 7

        # Dirt patches
        dirt_mask = (self.grid == 2) * (
            np.random.rand(self.size, self.size) < dirt_probability
        )
        self.grid[dirt_mask] = 4

    def generate_terrain(
        self,
        seed: int,
        water_threshold=0.35,
        snow_threshold=0.2,
        rock_probability=0.06,
        dirt_probability=0.05,
    ):
        """Generates the terrain for the world based on Perlin noise."""
        self._generate_noise(seed)
        # self._apply_biomes(water_threshold, snow_threshold)
        # self._apply_features(rock_probability, dirt_probability)

        for y in range(self.size):
            for x in range(self.size):
                noise_val = self.noise[y, x]
                # Water
                if noise_val <= water_threshold:
                    self.grid[y, x] = 0  # Deep Water
                    if (
                        noise_val > water_threshold - 0.05
                    ):  # Creates a shallow water band
                        self.grid[y, x] = 1  # Shallow Water
                # Land
                else:
                    self.grid[y, x] = 2  # Grass
                    # Add variation with probabilities
                    if np.random.rand() < rock_probability:
                        self.grid[y, x] = 7  # Rocks
                    elif np.random.rand() < dirt_probability:
                        self.grid[y, x] = 4  # Dirt
                    # Snow on high ground
                    if noise_val >= 1 - snow_threshold:
                        self.grid[y, x] = 5  # Snow

    def get_image_data(self):
        """
        Returns the world map as a color image.
        """
        height, width = self.grid.shape
        image_data = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(len(self.colors)):
            image_data[self.grid == i] = self.colors[i]
        return image_data

    def plot(self):
        """Plots the generated world grid using matplotlib."""
        image = self.get_image_data()
        plt.imshow(image, interpolation="nearest")
        plt.title("Procedural World Map")
        plt.xlabel("X-Coordinate")
        plt.ylabel("Y-Coordinate")

        handles = [
            mpatches.Patch(color=self.colors[0] / 255.0, label="Deep Water"),
            mpatches.Patch(color=self.colors[1] / 255.0, label="Shallow Water"),
            mpatches.Patch(color=self.colors[2] / 255.0, label="Grass"),
            mpatches.Patch(color=self.colors[3] / 255.0, label="Forest"),
            mpatches.Patch(color=self.colors[4] / 255.0, label="Dirt"),
            mpatches.Patch(color=self.colors[5] / 255.0, label="Snow"),
            mpatches.Patch(color=self.colors[6] / 255.0, label="Sand"),
            mpatches.Patch(color=self.colors[7] / 255.0, label="Rocks"),
        ]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

    def create_3d_mesh(self):
        """
        Generates vertices, triangles, colors, normals, and UVs for a 3D mesh.
        """
        if self.noise is None:
            raise ValueError("Noise data not generated. Call generate_terrain() first.")

        height_scale = 10
        size = self.size
        vertices_np = np.zeros((size * size, 3), dtype=np.float32)
        colors_np = np.zeros((size * size, 4), dtype=np.float32)  # RGBA format
        uvs_np = np.zeros((size * size, 2), dtype=np.float32)

        # Generate vertices and colors
        for y in range(size):
            for x in range(size):
                idx = y * size + x
                noise_val = self.noise[y, x]
                height = noise_val * height_scale
                vertices_np[idx] = [x, height, y]
                rgb_color = self.colors[self.grid[y, x]] / 255.0
                colors_np[idx] = [
                    rgb_color[0],
                    rgb_color[1],
                    rgb_color[2],
                    1.0,
                ]

        # Create triangles
        triangles_np = []
        for y in range(size - 1):
            for x in range(size - 1):
                v1 = y * size + x
                v2 = y * size + x + 1
                v3 = (y + 1) * size + x
                v4 = (y + 1) * size + x + 1
                # First triangle
                triangles_np.extend([v1, v3, v2])
                # Second triangle
                triangles_np.extend([v2, v3, v4])
        triangles_np = np.array(triangles_np, dtype=np.uint32)

        # Compute normals from vertices and triangles
        normals = np.zeros_like(vertices_np)
        for i in range(0, len(triangles_np), 3):
            v_a = vertices_np[triangles_np[i]]
            v_b = vertices_np[triangles_np[i + 1]]
            v_c = vertices_np[triangles_np[i + 2]]

            edge1 = v_b - v_a
            edge2 = v_c - v_a
            normal = np.cross(edge1, edge2)
            norm_val = np.linalg.norm(normal)
            if norm_val > 0:
                normal /= norm_val
            normals[triangles_np[i]] += normal
            normals[triangles_np[i + 1]] += normal
            normals[triangles_np[i + 2]] += normal

        normals_norm = np.sqrt(
            normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2
        )
        normals[normals_norm != 0] /= normals_norm[normals_norm != 0].reshape(-1, 1)

        # Compute UVs
        uvs = np.zeros((size * size, 2), dtype=np.float32)
        for y in range(size):
            for x in range(size):
                idx = y * size + x
                uvs[idx] = [x / size, y / size]

        return vertices_np, triangles_np, colors_np, normals, uvs


if __name__ == "__main__":
    world = WorldGenerator(size=256)
    world.generate_terrain(
        seed=42,
        water_threshold=0.3,
        snow_threshold=0.2,
        rock_probability=0.06,
        dirt_probability=0.05,
    )
    world.plot()
