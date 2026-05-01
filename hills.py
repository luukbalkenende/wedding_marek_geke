"""Hill field construction utilities."""

from typing import Sequence

import numpy as np


def lon_lat_to_normalized_xy(
    locations: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Convert longitude/latitude pairs to normalized map coordinates.

    Parameters
    ----------
    locations : Sequence[tuple[float, float]]
        Sequence of ``(longitude, latitude)`` pairs in degrees.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` containing normalized ``(x, y)`` in [0, 1].
    """
    coords = np.asarray(locations, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("locations must be a sequence of (longitude, latitude)")

    lon = coords[:, 0]
    lat = coords[:, 1]
    x = (lon + 180.0) / 360.0
    y = (90.0 - lat) / 180.0
    return np.column_stack((x, y))


def gaussian_height_map(
    image_shape: tuple[int, int, int],
    destination_xy: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Compute destination-based Gaussian height map.

    Parameters
    ----------
    image_shape : tuple[int, int, int]
        Shape of map image as ``(H, W, C)``.
    destination_xy : np.ndarray
        Destination points as normalized coordinates with shape ``(N, 2)``.
    sigma : float
        Gaussian spread in normalized map units.

    Returns
    -------
    np.ndarray
        Height map with shape ``(H, W)``.
    """
    height, width = image_shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = xx / (width - 1)
    y_norm = yy / (height - 1)

    heights = np.zeros((height, width), dtype=np.float64)
    for x0, y0 in destination_xy:
        dx = np.abs(x_norm - x0)
        dx = np.minimum(dx, 1.0 - dx)  # Wrap longitude around date line.
        dy = y_norm - y0
        dist_sq = dx * dx + dy * dy
        heights += np.exp(-dist_sq / (sigma * sigma))
    return heights


def normalize_height(heights: np.ndarray) -> np.ndarray:
    """Normalize a height map to [0, 1].

    Parameters
    ----------
    heights : np.ndarray
        Raw height field.

    Returns
    -------
    np.ndarray
        Normalized height field in [0, 1].
    """
    h_min = float(heights.min())
    h_max = float(heights.max())
    if h_max == h_min:
        return np.zeros_like(heights)
    return (heights - h_min) / (h_max - h_min)
