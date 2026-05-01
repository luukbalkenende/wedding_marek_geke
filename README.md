# World Hills

Create a hilly world visualization from destination points:

$$
\text{score}(x, y) = \sum_{d \in \text{destinations}} \exp\left(
-\frac{\text{distance}((x, y), d)^2}{\sigma^2}
\right)
$$

## Project Structure

- `main.py`  
  CLI entrypoint. Parses arguments and orchestrates the pipeline.

- `locations.py`  
  Default destination locations.

- `settings.py`  
  Global configuration values (sigma, dimensions, output names, texture URLs, cache directory).

- `hills.py`  
  Pure terrain logic:
  - lon/lat to normalized coordinates
  - Gaussian height map construction
  - height normalization

- `renderer.py`  
  Rendering and image I/O:
  - satellite texture fetch/cache/load
  - 3D hilly PNG generation
  - top-down height PNG generation
  - grid line overlay on terrain

## How To Run

```bash
python main.py
```

This writes:
- `world_hills.png`
- `world_hills_topdown.png`
- `world_hills_trajectory.png`
- `world_hills_topdown_trajectory.png`

## Common Options

```bash
python main.py \
  --sigma 0.075 \
  --vertical-scale 0.42 \
  --width 1440 \
  --height 720 \
  --output "world_hills.png" \
  --topdown-output "world_hills_topdown.png" \
  --trajectory-output "world_hills_trajectory.png" \
  --topdown-trajectory-output "world_hills_topdown_trajectory.png"
```

Provide custom destinations by repeating `--location`:

```bash
python main.py \
  --location "4.9,52.37" \
  --location "-74.0,40.71" \
  --location "100.50,13.76"
```

Format is `longitude,latitude` in degrees.

To use the curated `DESTINATIONS` list from `locations.py` (travel spots) instead of `DEFAULT_LOCATIONS`, run:

```bash
python main.py --destinations
```

Any `--location` you pass still wins over `--destinations`.

## Gradient Descent / Ascent Search

The app now runs a gradient-based search on the hill landscape (implemented as
gradient ascent on height, equivalent to gradient descent on negative height).

- Start point: highest point + slight random offset.
- Step: `learning_rate * gradient`.
- Stop: movement below threshold for consecutive iterations.

At runtime it prints:

`Worst location possible is found to be in: lon=..., lat=...`

Main optimizer flags:

- `--gd-learning-rate`
- `--gd-max-iters`
- `--gd-convergence-tol`
- `--gd-patience`
- `--gd-start-offset`
- `--gd-start-offset-north` (negative means south)
- `--gd-start-offset-east` (negative means west)
- `--gd-seed`

## Ball Rolling Simulation

Default behavior now uses a ball simulation mode (`--search-mode ball`):

- Start point: highest point + slight random offset.
- Physics: downhill acceleration from slope + damping/friction.
- Stop: speed stays below threshold for consecutive steps.

Main ball flags:

- `--search-mode ball`
- `--ball-dt`
- `--ball-gravity`
- `--ball-damping`
- `--ball-speed-tol`
- `--hilly-sharpen-amount` (0 disables sharpening)

To switch back to old gradient optimizer:

```bash
python main.py --search-mode gradient
```

## Skip Selected Outputs

Use these flags to skip expensive image renders:

- `--no-hilly-image`
- `--no-topdown-image`
- `--no-trajectory-image`
- `--no-topdown-trajectory-image`

Example (skip only the hilly 3D image):

```bash
python main.py --no-hilly-image
```
