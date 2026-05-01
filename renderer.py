"""Image creation and rendering utilities."""

from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def fetch_satellite_texture(
    cache_dir: str | Path,
    satellite_texture_urls: Sequence[str],
) -> Path:
    """Download and cache an equirectangular satellite texture.

    Parameters
    ----------
    cache_dir : str | Path
        Cache directory where the texture is stored.
    satellite_texture_urls : Sequence[str]
        Candidate texture URLs.

    Returns
    -------
    pathlib.Path
        Path to local cached texture.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    texture_file = cache_path / "earth_satellite_texture"
    extensions = [".jpg", ".jpeg", ".png", ".webp"]

    for ext in extensions:
        existing = texture_file.with_suffix(ext)
        if existing.exists():
            return existing

    last_error: Exception | None = None
    for url in satellite_texture_urls:
        suffix = Path(url).suffix.lower()
        ext = suffix if suffix in extensions else ".jpg"
        candidate = texture_file.with_suffix(ext)
        try:
            urlretrieve(url, candidate)
            return candidate
        except (HTTPError, URLError, ValueError) as exc:
            last_error = exc
            if candidate.exists():
                candidate.unlink()

    raise RuntimeError(
        "Failed to download a satellite texture from all fallback URLs."
    ) from last_error


def load_world_texture(
    width: int,
    height: int,
    cache_dir: str | Path,
    satellite_texture_urls: Sequence[str],
) -> np.ndarray:
    """Load world texture as normalized RGB image.

    Parameters
    ----------
    width : int
        Output width in pixels.
    height : int
        Output height in pixels.
    cache_dir : str | Path
        Cache directory for downloaded textures.
    satellite_texture_urls : Sequence[str]
        Candidate texture URLs.

    Returns
    -------
    np.ndarray
        RGB array with shape ``(H, W, 3)`` in [0, 1].
    """
    texture_file = fetch_satellite_texture(
        cache_dir=cache_dir,
        satellite_texture_urls=satellite_texture_urls,
    )
    image = Image.open(texture_file).convert("RGB")
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def draw_map_grid_lines(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    row_step: int = 18,
    col_step: int = 36,
) -> None:
    """Overlay yellow grid lines on terrain.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        3D axes where terrain is drawn.
    x : np.ndarray
        X coordinates of shape ``(H, W)``.
    y : np.ndarray
        Y coordinates of shape ``(H, W)``.
    z : np.ndarray
        Z coordinates of shape ``(H, W)``.
    row_step : int, optional
        Row stride for horizontal grid lines.
    col_step : int, optional
        Column stride for vertical grid lines.
    """
    ax.plot_surface(
        x,
        y,
        z + 1e-6,
        rstride=max(1, row_step),
        cstride=max(1, col_step),
        color=(1.0, 1.0, 1.0, 0.0),
        linewidth=0.75,
        edgecolor="#FFE066",
        antialiased=False,
        shade=False,
    )


def apply_unsharp_mask(image_rgb: np.ndarray, amount: float = 0.9) -> np.ndarray:
    """Sharpen an RGB image using a simple unsharp-mask kernel.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image in [0, 1] with shape ``(H, W, 3)``.
    amount : float, optional
        Sharpen strength. Use 0 for no sharpening.

    Returns
    -------
    np.ndarray
        Sharpened RGB image in [0, 1].
    """
    if amount <= 0.0:
        return image_rgb

    padded = np.pad(image_rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = (
        padded[0:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 2:]
        + (4.0 * padded[1:-1, 1:-1])
    ) / 8.0
    sharpened = image_rgb + amount * (image_rgb - blurred)
    return np.clip(sharpened, 0.0, 1.0)


def save_hilly_world_png(
    map_rgb: np.ndarray,
    heights_norm: np.ndarray,
    output_path: str | Path,
    vertical_scale: float,
    sharpen_amount: float = 0.9,
    elev: float = 45.0,
    azim: float = -90.0,
) -> None:
    """Save 3D hilly world PNG.

    Parameters
    ----------
    map_rgb : np.ndarray
        RGB texture array of shape ``(H, W, 3)``.
    heights_norm : np.ndarray
        Normalized height array of shape ``(H, W)``.
    output_path : str | Path
        Path to save PNG.
    vertical_scale : float
        Scale multiplier for terrain height.
    sharpen_amount : float, optional
        Texture edge enhancement strength for hilly render.
    elev : float, optional
        Camera elevation in degrees.
    azim : float, optional
        Camera azimuth in degrees.
    """
    height, width = heights_norm.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / (width - 1)
    y = yy / (height - 1)
    z = np.power(heights_norm, 0.75) * vertical_scale
    texture = apply_unsharp_mask(map_rgb, amount=sharpen_amount)

    fig = plt.figure(figsize=(14, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=texture,
        linewidth=0.0,
        edgecolor="none",
        antialiased=False,
        alpha=1.0,
        shade=False,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_zlim(0.0, vertical_scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((2.0, 1.0, 0.25))
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_topdown_height_png(
    map_rgb: np.ndarray,
    heights_norm: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save top-down height-overlay PNG.

    Parameters
    ----------
    map_rgb : np.ndarray
        RGB texture array of shape ``(H, W, 3)``.
    heights_norm : np.ndarray
        Normalized height array of shape ``(H, W)``.
    output_path : str | Path
        Path to save top-down PNG.
    """
    height, width = heights_norm.shape
    fig = plt.figure(figsize=(14, 7), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(map_rgb, interpolation="bilinear")
    ax.imshow(
        heights_norm,
        cmap="magma",
        alpha=np.clip(0.18 + 0.65 * heights_norm, 0.18, 0.83),
        interpolation="bilinear",
    )
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_hilly_world_with_trajectory_png(
    map_rgb: np.ndarray,
    heights_norm: np.ndarray,
    trajectory_xy: np.ndarray,
    output_path: str | Path,
    vertical_scale: float,
    sharpen_amount: float = 0.9,
    elev: float = 45.0,
    azim: float = -90.0,
) -> None:
    """Save 3D hilly world with trajectory embedded in texture.

    Parameters
    ----------
    map_rgb : np.ndarray
        RGB texture array of shape ``(H, W, 3)``.
    heights_norm : np.ndarray
        Normalized height array of shape ``(H, W)``.
    trajectory_xy : np.ndarray
        Optimization trajectory in normalized coordinates ``(N, 2)``.
    output_path : str | Path
        Path to save PNG.
    vertical_scale : float
        Scale multiplier for terrain height.
    sharpen_amount : float, optional
        Texture edge enhancement strength for hilly render.
    elev : float, optional
        Camera elevation in degrees.
    azim : float, optional
        Camera azimuth in degrees.
    """
    height, width = heights_norm.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / (width - 1)
    y = yy / (height - 1)
    z = np.power(heights_norm, 0.75) * vertical_scale

    # 1-3) Draw trajectory in 2D map space.
    overlay_fig = plt.figure(figsize=(width / 200, height / 200), dpi=200)
    overlay_ax = overlay_fig.add_axes([0, 0, 1, 1])
    base_texture = apply_unsharp_mask(map_rgb, amount=sharpen_amount)
    overlay_ax.imshow(base_texture, interpolation="bilinear")
    # Keep every other point only for this hilly-trajectory render.
    trajectory_sparse = trajectory_xy[::2]
    if trajectory_sparse.shape[0] < 2 and trajectory_xy.shape[0] >= 2:
        trajectory_sparse = trajectory_xy[[0, -1]]

    traj_x_px = np.mod(trajectory_sparse[:, 0], 1.0) * (width - 1)
    traj_y_px = np.clip(trajectory_sparse[:, 1], 0.0, 1.0) * (height - 1)
    overlay_ax.plot(traj_x_px, traj_y_px, color="red", linewidth=1.1, alpha=1.0)
    overlay_ax.scatter(
        traj_x_px,
        traj_y_px,
        color="red",
        marker="x",
        s=14,
        linewidths=0.9,
    )
    overlay_ax.set_xlim(0, width - 1)
    overlay_ax.set_ylim(height - 1, 0)
    overlay_ax.axis("off")
    overlay_fig.canvas.draw()
    overlay_buffer = np.frombuffer(overlay_fig.canvas.buffer_rgba(), dtype=np.uint8)
    textured_with_path = (
        overlay_buffer.reshape(height, width, 4)[:, :, :3].astype(np.float32) / 255.0
    )
    plt.close(overlay_fig)

    # 4-5) Distort textured map via hilly 3D surface and save.
    fig = plt.figure(figsize=(14, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=textured_with_path,
        linewidth=0.0,
        edgecolor="none",
        antialiased=False,
        alpha=1.0,
        shade=False,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_zlim(0.0, vertical_scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((2.0, 1.0, 0.25))
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_topdown_with_trajectory_png(
    map_rgb: np.ndarray,
    heights_norm: np.ndarray,
    trajectory_xy: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save top-down map with gradient trajectory overlay.

    Parameters
    ----------
    map_rgb : np.ndarray
        RGB texture array of shape ``(H, W, 3)``.
    heights_norm : np.ndarray
        Normalized height array of shape ``(H, W)``.
    trajectory_xy : np.ndarray
        Optimization trajectory in normalized coordinates ``(N, 2)``.
    output_path : str | Path
        Path to save top-down trajectory PNG.
    """
    height, width = heights_norm.shape
    x_pixels = np.mod(trajectory_xy[:, 0], 1.0) * (width - 1)
    y_pixels = np.clip(trajectory_xy[:, 1], 0.0, 1.0) * (height - 1)

    fig = plt.figure(figsize=(14, 7), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(map_rgb, interpolation="bilinear")
    ax.imshow(
        heights_norm,
        cmap="magma",
        alpha=np.clip(0.18 + 0.65 * heights_norm, 0.18, 0.83),
        interpolation="bilinear",
    )
    ax.plot(x_pixels, y_pixels, color="red", linewidth=1.7, alpha=1.0)
    ax.scatter(x_pixels, y_pixels, color="red", marker="x", s=22, linewidths=1.2)
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
