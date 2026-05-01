"""Search and simulation methods on the hill landscape."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GradientAscentResult:
    """Container for optimization result."""

    best_xy: np.ndarray
    best_lon_lat: tuple[float, float]
    trajectory_xy: np.ndarray
    iterations: int


def normalized_xy_to_lon_lat(xy: np.ndarray) -> tuple[float, float]:
    """Convert normalized map coordinate to lon/lat.

    Parameters
    ----------
    xy : np.ndarray
        Point in normalized map coordinates ``[x, y]``.

    Returns
    -------
    tuple[float, float]
        ``(longitude, latitude)`` in degrees.
    """
    lon = (float(xy[0]) * 360.0) - 180.0
    lat = 90.0 - (float(xy[1]) * 180.0)
    return lon, lat


def _bilinear_sample(field: np.ndarray, x: float, y: float) -> float:
    """Sample 2D field at normalized coordinate with bilinear interpolation."""
    height, width = field.shape
    x = float(np.mod(x, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    px = x * (width - 1)
    py = y * (height - 1)
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = px - x0
    ty = py - y0
    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]
    top = (1.0 - tx) * v00 + tx * v10
    bottom = (1.0 - tx) * v01 + tx * v11
    return float((1.0 - ty) * top + ty * bottom)


def run_gradient_ascent(
    heights_norm: np.ndarray,
    learning_rate: float,
    max_iters: int,
    convergence_tol: float,
    patience: int,
    start_offset_scale: float,
    start_offset_north: float,
    start_offset_east: float,
    rng_seed: int | None = None,
) -> GradientAscentResult:
    """Run gradient ascent to find highest point in hilly landscape.

    Parameters
    ----------
    heights_norm : np.ndarray
        Normalized height field in [0, 1] with shape ``(H, W)``.
    learning_rate : float
        Step-size multiplier.
    max_iters : int
        Maximum number of optimization steps.
    convergence_tol : float
        Movement tolerance threshold for convergence.
    patience : int
        Number of consecutive small movements before stopping.
    start_offset_scale : float
        Random offset scale around global max, in normalized coordinates.
    start_offset_north : float
        Deterministic north/south nudge from highest point. Positive moves north,
        negative moves south.
    start_offset_east : float
        Deterministic east/west nudge from highest point. Positive moves east,
        negative moves west.
    rng_seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    GradientAscentResult
        Optimization result including final coordinate and trajectory.
    """
    rng = np.random.default_rng(rng_seed)
    max_idx = np.unravel_index(int(np.argmax(heights_norm)), heights_norm.shape)
    max_y = max_idx[0] / (heights_norm.shape[0] - 1)
    max_x = max_idx[1] / (heights_norm.shape[1] - 1)
    start = np.array([max_x, max_y], dtype=np.float64)
    start[0] += start_offset_east
    start[1] -= start_offset_north
    start += rng.normal(0.0, start_offset_scale, size=2)
    start[0] = np.mod(start[0], 1.0)
    start[1] = np.clip(start[1], 0.0, 1.0)

    y_coords = np.linspace(0.0, 1.0, heights_norm.shape[0], dtype=np.float64)
    x_coords = np.linspace(0.0, 1.0, heights_norm.shape[1], dtype=np.float64)
    grad_y, grad_x = np.gradient(heights_norm, y_coords, x_coords)

    xy = start.copy()
    trajectory: list[np.ndarray] = [xy.copy()]
    small_move_count = 0
    iterations = 0

    for step in range(max_iters):
        gx = _bilinear_sample(grad_x, xy[0], xy[1])
        gy = _bilinear_sample(grad_y, xy[0], xy[1])
        grad = np.array([gx, gy], dtype=np.float64)
        delta = learning_rate * grad
        xy = xy + delta
        xy[0] = np.mod(xy[0], 1.0)
        xy[1] = np.clip(xy[1], 0.0, 1.0)
        trajectory.append(xy.copy())
        iterations = step + 1

        if float(np.linalg.norm(delta)) < convergence_tol:
            small_move_count += 1
        else:
            small_move_count = 0
        if small_move_count >= patience:
            break

    best_lon_lat = normalized_xy_to_lon_lat(xy)
    return GradientAscentResult(
        best_xy=xy,
        best_lon_lat=best_lon_lat,
        trajectory_xy=np.asarray(trajectory, dtype=np.float64),
        iterations=iterations,
    )


def run_ball_roll(
    heights_norm: np.ndarray,
    max_iters: int,
    patience: int,
    start_offset_scale: float,
    start_offset_north: float,
    start_offset_east: float,
    dt: float,
    gravity: float,
    damping: float,
    speed_tol: float,
    rng_seed: int | None = None,
) -> GradientAscentResult:
    """Simulate a rolling ball on the height field until it settles.

    Parameters
    ----------
    heights_norm : np.ndarray
        Normalized height field in [0, 1] with shape ``(H, W)``.
    max_iters : int
        Maximum simulation steps.
    patience : int
        Number of consecutive low-speed steps before stopping.
    start_offset_scale : float
        Random offset scale around global max, in normalized coordinates.
    start_offset_north : float
        Deterministic north/south nudge from highest point. Positive moves north,
        negative moves south.
    start_offset_east : float
        Deterministic east/west nudge from highest point. Positive moves east,
        negative moves west.
    dt : float
        Simulation time step.
    gravity : float
        Strength of gravity acceleration along slope.
    damping : float
        Velocity damping coefficient.
    speed_tol : float
        Speed threshold for convergence.
    rng_seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    GradientAscentResult
        Simulation result including final coordinate and trajectory.
    """
    rng = np.random.default_rng(rng_seed)
    max_idx = np.unravel_index(int(np.argmax(heights_norm)), heights_norm.shape)
    max_y = max_idx[0] / (heights_norm.shape[0] - 1)
    max_x = max_idx[1] / (heights_norm.shape[1] - 1)
    xy = np.array([max_x, max_y], dtype=np.float64)
    xy[0] += start_offset_east
    xy[1] -= start_offset_north
    xy += rng.normal(0.0, start_offset_scale, size=2)
    xy[0] = np.mod(xy[0], 1.0)
    xy[1] = np.clip(xy[1], 0.0, 1.0)

    y_coords = np.linspace(0.0, 1.0, heights_norm.shape[0], dtype=np.float64)
    x_coords = np.linspace(0.0, 1.0, heights_norm.shape[1], dtype=np.float64)
    grad_y, grad_x = np.gradient(heights_norm, y_coords, x_coords)

    velocity = np.zeros(2, dtype=np.float64)
    trajectory: list[np.ndarray] = [xy.copy()]
    low_speed_count = 0
    iterations = 0

    for step in range(max_iters):
        gx = _bilinear_sample(grad_x, xy[0], xy[1])
        gy = _bilinear_sample(grad_y, xy[0], xy[1])
        # Ball rolls downhill: accelerate opposite to gradient.
        acceleration = (-gravity * np.array([gx, gy])) - (damping * velocity)
        velocity = velocity + (dt * acceleration)
        xy = xy + (dt * velocity)
        xy[0] = np.mod(xy[0], 1.0)
        xy[1] = np.clip(xy[1], 0.0, 1.0)
        trajectory.append(xy.copy())
        iterations = step + 1

        speed = float(np.linalg.norm(velocity))
        if speed < speed_tol:
            low_speed_count += 1
        else:
            low_speed_count = 0
        if low_speed_count >= patience:
            break

    best_lon_lat = normalized_xy_to_lon_lat(xy)
    return GradientAscentResult(
        best_xy=xy,
        best_lon_lat=best_lon_lat,
        trajectory_xy=np.asarray(trajectory, dtype=np.float64),
        iterations=iterations,
    )
