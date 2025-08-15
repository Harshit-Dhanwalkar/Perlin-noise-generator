from typing import Tuple

import numpy as np


def generate_perlin_noise_2d(shape: Tuple[int, int], res: Tuple[int, int]):
    """
    Generates 2D Perlin noise.

    Args:
        shape: The shape of the output noise array (height, width).
        res: The resolution of the noise grid (rows, columns).

    Returns:
        A 2D NumPy array of Perlin noise.
    """
    if not (shape[0] % res[0] == 0 and shape[1] % res[1] == 0):
        raise ValueError("Shape must be divisible by resolution.")

    # Create a grid of random gradient vectors
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    # Create coordinate grid
    x = np.linspace(0, res[0], shape[0], endpoint=False)
    y = np.linspace(0, res[1], shape[1], endpoint=False)
    coords = np.stack(np.meshgrid(y, x), axis=-1)

    # Get integer and fractional parts of coordinates
    coords_int = coords.astype(int)
    coords_frac = coords - coords_int

    # Get the four gradient vectors for each point
    g00 = gradients[coords_int[..., 0], coords_int[..., 1]]
    g10 = gradients[coords_int[..., 0] + 1, coords_int[..., 1]]
    g01 = gradients[coords_int[..., 0], coords_int[..., 1] + 1]
    g11 = gradients[coords_int[..., 0] + 1, coords_int[..., 1] + 1]

    # Calculate dot products
    def dot(g, x, y):
        return g[..., 0] * x + g[..., 1] * y

    n00 = dot(g00, coords_frac[..., 0], coords_frac[..., 1])
    n10 = dot(g10, coords_frac[..., 0] - 1, coords_frac[..., 1])
    n01 = dot(g01, coords_frac[..., 0], coords_frac[..., 1] - 1)
    n11 = dot(g11, coords_frac[..., 0] - 1, coords_frac[..., 1] - 1)

    # Interpolation
    def interpolate(a, b, t):
        return a + (b - a) * (3 - 2 * t) * t * t

    ix0 = interpolate(n00, n10, coords_frac[..., 0])
    ix1 = interpolate(n01, n11, coords_frac[..., 0])

    return np.sqrt(2) * interpolate(ix0, ix1, coords_frac[..., 1])


def generate_fractal_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
):
    """
    Generates 2D fractal Perlin noise.

    Args:
        shape: The shape of the output noise array (height, width).
        res: The base resolution of the noise grid (rows, columns).
        octaves: The number of layers of noise to combine.
        persistence: The factor by which amplitude decreases for each octave.

    Returns:
        A 2D NumPy array of fractal Perlin noise.
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency * res[0], frequency * res[1])
        )
        frequency *= 2
        amplitude *= persistence
    return noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)
    noise = generate_perlin_noise_2d((256, 256), (8, 8))
    plt.imshow(noise, cmap="gray", interpolation="lanczos")
    plt.colorbar()

    np.random.seed(0)
    noise = generate_fractal_noise_2d((256, 256), (8, 8), 5)
    plt.figure()
    plt.imshow(noise, cmap="gray", interpolation="lanczos")
    plt.colorbar()
    plt.show()
