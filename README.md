# World Hills

**Finding the worst place to head for outdoor activities**—or at least playing with that idea visually on a globe.

You give a list of spots (honeymoon hikes, beaches, mountaintops, whatever counts as “the good stuff”). Each one adds a smooth bump on the world. Terrain is **high** near those places and lower elsewhere. From there you run a **gradient-descent style optimization** on that landscape, trace the path, and save renders so you can see where AI has determined for you to **not** go.

(However, if you are more into physics, just use the simulation of a ball rolling downhill).

## How it works

1. **Destinations** — Default lists live in `locations.py` (`DEFAULT_LOCATIONS`, or `DESTINATIONS` with `--destinations`). You can also pass any points as `--location lon,lat`.
2. **Score field** — For every map position \((x, y)\) in normalized coordinates, the height is built from Gaussian bumps centered on those destinations (width controlled by \(\sigma\)):

$$
\text{score}(x, y) = \sum_{d \in \text{destinations}} \exp\left(
-\frac{\text{distance}((x, y), d)^2}{\sigma^2}
\right)
$$

3. **Texture** — A cloud-free, equirectangular Earth image is fetched (cached under `.cache/`) and aligned with longitude/latitude.
4. **Hills & heatmaps** — That score becomes elevation for a **3D hilly globe** and a **top-down** view with height overlaid as color.
5. **Search path** — **Ball mode** (default): start near the global maximum with a small nudge, integrate velocity with gravity along the slope plus damping until the ball settles. **Gradient mode** (`--search-mode gradient`): gradient **descent** on the height field (steps opposite the gradient toward a low point / valley). The trajectory is drawn on trajectory PNGs and the final `(lon, lat)` is printed.
6. **Outputs** — By default everything is written under `output/` (filenames match `settings.py`, e.g. `output/test.png` unless you override paths).

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
pip install -r requirements.txt
python main.py
```

By default this writes (paths are configurable; these match `settings.py`):

- `output/test.png` — 3D hilly globe  
- `output/test_topdown.png` — top-down map + height heat  
- `output/test_trajectory.png` — same 3D view with path drawn on the texture  
- `output/test_topdown_trajectory.png` — top-down with path overlaid  

## Common Options

```bash
python main.py \
  --sigma 0.075 \
  --vertical-scale 0.42 \
  --width 1440 \
  --height 720 \
  --output "output/my_hills.png" \
  --topdown-output "output/my_topdown.png" \
  --trajectory-output "output/my_trajectory.png" \
  --topdown-trajectory-output "output/my_topdown_trajectory.png"
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

## Gradient Descent Search (`--search-mode gradient`)

Gradient descent on the **height** landscape: each step moves opposite the gradient
toward a **local minimum** (valley), i.e. lower score / further from the Gaussian
peaks around your destinations (modulo which basin you land in).

- Start point: near global maximum + slight random/directional offset.
- Step: `-learning_rate * gradient(height)`.
- Stop: movement norm below threshold for consecutive iterations.

At runtime it prints the end point, e.g. **`Ball settles at location: lon=..., lat=...`** (ball mode) or **`Lowest terrain point reached near: lon=..., lat=...`** (gradient mode).

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

To use gradient descent instead of the ball:

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
