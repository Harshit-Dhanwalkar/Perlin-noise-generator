from typing import Tuple

import numpy as np
from numba import njit


@njit
def interpolant(t: np.ndarray):
    """
    Returns a smoother interpolation value.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


@njit
def generate_perlin_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    tileable: Tuple[bool, bool] = (False, False),
    interpolant=interpolant,
):
    """
    Generate a 2D numpy array of Perlin noise.
    """
    if not (shape[0] % res[0] == 0 and shape[1] % res[1] == 0):
        raise ValueError("Shape must be a multiple of res.")

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    xvals = np.arange(0, res[0], delta[0])
    yvals = np.arange(0, res[1], delta[1])
    grid = np.empty((2, len(xvals), len(yvals)))
    for j, y in enumerate(yvals):
        for i, x in enumerate(xvals):
            grid[0][i, j] = x
    yy = np.empty((len(xvals), len(yvals)))
    for i, x in enumerate(xvals):
        grid[1][i, :] = yvals
    grid = grid.transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    grad_matrix = np.empty((d[0] * gradients.shape[0], d[1] * gradients.shape[1], 2))
    for i in range(gradients.shape[0]):
        for j in range(gradients.shape[1]):
            grad_matrix[i * d[0] : (i + 1) * d[0], j * d[1] : (j + 1) * d[1]] = (
                gradients[i, j]
            )
    gradients = grad_matrix

    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    t1 = (1 - t[:, :, 1]) * n0
    t2 = t[:, :, 1] * n1
    sum_t = t1 + t2
    mult_s2 = np.sqrt(2) * sum_t
    return mult_s2


@njit
def generate_fractal_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: int = 2,
    tileable: Tuple[bool, bool] = (False, False),
    interpolant=interpolant,
):
    """
    Generate a 2D numpy array of fractal noise.
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency * res[0], frequency * res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise
