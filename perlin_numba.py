# perlin_numba.py
from typing import Tuple

import numpy as np
from numba import njit


@njit(fastmath=True)
def generate_perlin_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    tileable: Tuple[bool, bool] = (False, False),
):
    """
    Generate a 2D numpy array of Perlin noise with Numba.
    """

    def interpolant(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Gradients
    gradients = np.random.rand(res[0] + 1, res[1] + 1, 2) * 2 - 1
    norm = np.sqrt(gradients[..., 0] ** 2 + gradients[..., 1] ** 2)
    gradients[..., 0] /= norm
    gradients[..., 1] /= norm

    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]

    # Coordinate grid
    x_coords = np.arange(0, res[0], res[0] / shape[0])
    y_coords = np.arange(0, res[1], res[1] / shape[1])
    coords = np.zeros((shape[0], shape[1], 2))
    for i in range(shape[0]):
        for j in range(shape[1]):
            coords[i, j, 0] = y_coords[j]  # x coordinate
            coords[i, j, 1] = x_coords[i]  # y coordinate

    coords_int = coords.astype(np.int32)
    coords_frac = coords - coords_int

    # Calculate dot products
    n00 = np.zeros(shape)
    n10 = np.zeros(shape)
    n01 = np.zeros(shape)
    n11 = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            x_idx = coords_int[i, j, 0]
            y_idx = coords_int[i, j, 1]

            n00[i, j] = (
                gradients[x_idx, y_idx, 0] * coords_frac[i, j, 0]
                + gradients[x_idx, y_idx, 1] * coords_frac[i, j, 1]
            )
            n10[i, j] = (
                gradients[x_idx + 1, y_idx, 0] * (coords_frac[i, j, 0] - 1)
                + gradients[x_idx + 1, y_idx, 1] * coords_frac[i, j, 1]
            )
            n01[i, j] = gradients[x_idx, y_idx + 1, 0] * coords_frac[
                i, j, 0
            ] + gradients[x_idx, y_idx + 1, 1] * (coords_frac[i, j, 1] - 1)
            n11[i, j] = gradients[x_idx + 1, y_idx + 1, 0] * (
                coords_frac[i, j, 0] - 1
            ) + gradients[x_idx + 1, y_idx + 1, 1] * (coords_frac[i, j, 1] - 1)

    t_x = interpolant(coords_frac[..., 0])
    t_y = interpolant(coords_frac[..., 1])

    ix0 = n00 * (1 - t_x) + n10 * t_x
    ix1 = n01 * (1 - t_x) + n11 * t_x

    return np.sqrt(2) * (ix0 * (1 - t_y) + ix1 * t_y)


@njit(fastmath=True)
def generate_fractal_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: int = 2,
    tileable: Tuple[bool, bool] = (False, False),
):
    """
    Generate a 2D numpy array of fractal noise.
    """
    noise = np.zeros(shape)
    frequency = 1.0
    amplitude = 1.0
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (int(frequency * res[0]), int(frequency * res[1])), tileable
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise
