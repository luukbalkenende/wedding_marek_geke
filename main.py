"""Main CLI entry point for world hills generation."""

import argparse
from pathlib import Path
from typing import Iterable

from hills import gaussian_height_map, lon_lat_to_normalized_xy, normalize_height
from locations import DEFAULT_LOCATIONS, DESTINATIONS
from optimizer import run_ball_roll, run_gradient_descent
from renderer import (
    load_world_texture,
    save_hilly_world_png,
    save_hilly_world_with_trajectory_png,
    save_topdown_with_trajectory_png,
    save_topdown_height_png,
)
from settings import (
    DEFAULT_CACHE_DIR,
    DEFAULT_GD_CONVERGENCE_TOL,
    DEFAULT_GD_LEARNING_RATE,
    DEFAULT_GD_MAX_ITERS,
    DEFAULT_GD_PATIENCE,
    DEFAULT_GD_START_OFFSET,
    DEFAULT_GD_START_OFFSET_EAST,
    DEFAULT_GD_START_OFFSET_NORTH,
    DEFAULT_HEIGHT,
    DEFAULT_OUTPUT,
    DEFAULT_SATELLITE_TEXTURE_URLS,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SIGMA,
    DEFAULT_TOPDOWN_OUTPUT,
    DEFAULT_TOPDOWN_TRAJECTORY_OUTPUT,
    DEFAULT_TRAJECTORY_OUTPUT,
    DEFAULT_VERTICAL_SCALE,
    DEFAULT_WIDTH,
    DEFAULT_BALL_DAMPING,
    DEFAULT_BALL_DT,
    DEFAULT_BALL_GRAVITY,
    DEFAULT_BALL_SPEED_TOL,
    DEFAULT_HILLY_SHARPEN_AMOUNT,
)


def parse_locations(values: Iterable[str]) -> list[tuple[float, float]]:
    """Parse CLI longitude/latitude pairs.

    Parameters
    ----------
    values : Iterable[str]
        Items formatted as ``"lon,lat"``, for example ``"4.9,52.37"``.

    Returns
    -------
    list[tuple[float, float]]
        Parsed destination list in degrees.
    """
    points: list[tuple[float, float]] = []
    for item in values:
        try:
            lon_str, lat_str = item.split(",")
            points.append((float(lon_str), float(lat_str)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid location '{item}'. Use lon,lat for each point."
            ) from exc
    return points


def build_arg_parser() -> argparse.ArgumentParser:
    """Create command line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser instance for CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Render a world map as a hilly landscape."
    )
    parser.add_argument(
        "--location",
        action="append",
        default=[],
        help=(
            "Destination lon,lat in degrees. Repeat for multiple points. "
            "If given, overrides --destinations."
        ),
    )
    parser.add_argument(
        "--destinations",
        action="store_true",
        help=(
            "Use locations.DESTINATIONS instead of DEFAULT_LOCATIONS when "
            "no --location is passed."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"Gaussian width in normalized map units (default: {DEFAULT_SIGMA}).",
    )
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=DEFAULT_VERTICAL_SCALE,
        help=f"Height multiplier for 3D rendering (default: {DEFAULT_VERTICAL_SCALE}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Texture width in pixels (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Texture height in pixels (default: {DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"3D output PNG path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--topdown-output",
        type=str,
        default=DEFAULT_TOPDOWN_OUTPUT,
        help=f"Top-down output PNG path (default: {DEFAULT_TOPDOWN_OUTPUT}).",
    )
    parser.add_argument(
        "--trajectory-output",
        type=str,
        default=DEFAULT_TRAJECTORY_OUTPUT,
        help=f"Trajectory output PNG path (default: {DEFAULT_TRAJECTORY_OUTPUT}).",
    )
    parser.add_argument(
        "--topdown-trajectory-output",
        type=str,
        default=DEFAULT_TOPDOWN_TRAJECTORY_OUTPUT,
        help=(
            "Top-down trajectory output PNG path "
            f"(default: {DEFAULT_TOPDOWN_TRAJECTORY_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--gd-learning-rate",
        type=float,
        default=DEFAULT_GD_LEARNING_RATE,
        help=f"Gradient descent learning rate (default: {DEFAULT_GD_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--gd-max-iters",
        type=int,
        default=DEFAULT_GD_MAX_ITERS,
        help=f"Gradient descent max iterations (default: {DEFAULT_GD_MAX_ITERS}).",
    )
    parser.add_argument(
        "--gd-convergence-tol",
        type=float,
        default=DEFAULT_GD_CONVERGENCE_TOL,
        help=(
            "Stop threshold for small movement "
            f"(default: {DEFAULT_GD_CONVERGENCE_TOL})."
        ),
    )
    parser.add_argument(
        "--gd-patience",
        type=int,
        default=DEFAULT_GD_PATIENCE,
        help=f"Consecutive small-step patience (default: {DEFAULT_GD_PATIENCE}).",
    )
    parser.add_argument(
        "--gd-start-offset",
        type=float,
        default=DEFAULT_GD_START_OFFSET,
        help=(
            "Random offset scale around highest point "
            f"(default: {DEFAULT_GD_START_OFFSET})."
        ),
    )
    parser.add_argument(
        "--gd-start-offset-north",
        type=float,
        default=DEFAULT_GD_START_OFFSET_NORTH,
        help=(
            "Deterministic north/south start nudge from highest point "
            f"(default: {DEFAULT_GD_START_OFFSET_NORTH}). "
            "Positive is north, negative is south."
        ),
    )
    parser.add_argument(
        "--gd-start-offset-east",
        type=float,
        default=DEFAULT_GD_START_OFFSET_EAST,
        help=(
            "Deterministic east/west start nudge from highest point "
            f"(default: {DEFAULT_GD_START_OFFSET_EAST}). "
            "Positive is east, negative is west."
        ),
    )
    parser.add_argument(
        "--gd-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible optimizer start.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["ball", "gradient"],
        default=DEFAULT_SEARCH_MODE,
        help=f"Trajectory search method (default: {DEFAULT_SEARCH_MODE}).",
    )
    parser.add_argument(
        "--ball-dt",
        type=float,
        default=DEFAULT_BALL_DT,
        help=f"Ball simulation time step (default: {DEFAULT_BALL_DT}).",
    )
    parser.add_argument(
        "--ball-gravity",
        type=float,
        default=DEFAULT_BALL_GRAVITY,
        help=f"Ball simulation gravity factor (default: {DEFAULT_BALL_GRAVITY}).",
    )
    parser.add_argument(
        "--ball-damping",
        type=float,
        default=DEFAULT_BALL_DAMPING,
        help=f"Ball simulation damping factor (default: {DEFAULT_BALL_DAMPING}).",
    )
    parser.add_argument(
        "--ball-speed-tol",
        type=float,
        default=DEFAULT_BALL_SPEED_TOL,
        help=f"Ball convergence speed threshold (default: {DEFAULT_BALL_SPEED_TOL}).",
    )
    parser.add_argument(
        "--hilly-sharpen-amount",
        type=float,
        default=DEFAULT_HILLY_SHARPEN_AMOUNT,
        help=(
            "Edge enhancement amount for hilly renders "
            f"(default: {DEFAULT_HILLY_SHARPEN_AMOUNT})."
        ),
    )
    parser.add_argument(
        "--no-hilly-image",
        action="store_true",
        help="Skip writing the main 3D hilly image.",
    )
    parser.add_argument(
        "--no-topdown-image",
        action="store_true",
        help="Skip writing the top-down heat overlay image.",
    )
    parser.add_argument(
        "--no-trajectory-image",
        action="store_true",
        help="Skip writing the 3D trajectory overlay image.",
    )
    parser.add_argument(
        "--no-topdown-trajectory-image",
        action="store_true",
        help="Skip writing the top-down trajectory overlay image.",
    )
    return parser


def resolve_output_path(path_value: str) -> Path:
    """Resolve output path, defaulting plain filenames to output/ directory.

    Parameters
    ----------
    path_value : str
        Requested output path from CLI.

    Returns
    -------
    pathlib.Path
        Resolved path with parent directory ensured later in pipeline.
    """
    requested = Path(path_value)
    if requested.parent == Path("."):
        return Path("output") / requested
    return requested


def main() -> None:
    """Run world hills generation pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.location:
        locations = parse_locations(args.location)
    elif args.destinations:
        locations = list(DESTINATIONS)
    else:
        locations = list(DEFAULT_LOCATIONS)

    if args.width < 200 or args.height < 100:
        raise ValueError("--width and --height are too small for a world render.")

    map_rgb = load_world_texture(
        width=args.width,
        height=args.height,
        cache_dir=DEFAULT_CACHE_DIR,
        satellite_texture_urls=DEFAULT_SATELLITE_TEXTURE_URLS,
    )
    destination_xy = lon_lat_to_normalized_xy(locations)
    heights = gaussian_height_map(map_rgb.shape, destination_xy, sigma=args.sigma)
    heights_norm = normalize_height(heights)

    output_path = resolve_output_path(args.output)
    topdown_output_path = resolve_output_path(args.topdown_output)
    trajectory_output_path = resolve_output_path(args.trajectory_output)
    topdown_trajectory_output_path = resolve_output_path(args.topdown_trajectory_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_output_path.parent.mkdir(parents=True, exist_ok=True)
    topdown_trajectory_output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.search_mode == "ball":
        ascent_result = run_ball_roll(
            heights_norm=heights_norm,
            max_iters=args.gd_max_iters,
            patience=args.gd_patience,
            start_offset_scale=args.gd_start_offset,
            start_offset_north=args.gd_start_offset_north,
            start_offset_east=args.gd_start_offset_east,
            dt=args.ball_dt,
            gravity=args.ball_gravity,
            damping=args.ball_damping,
            speed_tol=args.ball_speed_tol,
            rng_seed=args.gd_seed,
        )
    else:
        ascent_result = run_gradient_descent(
            heights_norm=heights_norm,
            learning_rate=args.gd_learning_rate,
            max_iters=args.gd_max_iters,
            convergence_tol=args.gd_convergence_tol,
            patience=args.gd_patience,
            start_offset_scale=args.gd_start_offset,
            start_offset_north=args.gd_start_offset_north,
            start_offset_east=args.gd_start_offset_east,
            rng_seed=args.gd_seed,
        )
    lon, lat = ascent_result.best_lon_lat
    if args.search_mode == "ball":
        print(
            "Ball settles at location: "
            f"lon={lon:.4f}, lat={lat:.4f} "
            f"(steps={ascent_result.iterations})"
        )
    else:
        print(
            "Lowest terrain point reached near: "
            f"lon={lon:.4f}, lat={lat:.4f} "
            f"(iterations={ascent_result.iterations})"
        )

    if not args.no_hilly_image:
        save_hilly_world_png(
            map_rgb=map_rgb,
            heights_norm=heights_norm,
            output_path=output_path,
            vertical_scale=args.vertical_scale,
            sharpen_amount=args.hilly_sharpen_amount,
        )
    if not args.no_topdown_image:
        save_topdown_height_png(
            map_rgb=map_rgb,
            heights_norm=heights_norm,
            output_path=topdown_output_path,
        )
    if not args.no_trajectory_image:
        save_hilly_world_with_trajectory_png(
            map_rgb=map_rgb,
            heights_norm=heights_norm,
            trajectory_xy=ascent_result.trajectory_xy,
            output_path=trajectory_output_path,
            vertical_scale=args.vertical_scale,
            sharpen_amount=args.hilly_sharpen_amount,
        )
    if not args.no_topdown_trajectory_image:
        save_topdown_with_trajectory_png(
            map_rgb=map_rgb,
            heights_norm=heights_norm,
            trajectory_xy=ascent_result.trajectory_xy,
            output_path=topdown_trajectory_output_path,
        )


if __name__ == "__main__":
    main()
