"""Configuration values for world hill rendering."""

DEFAULT_SATELLITE_TEXTURE_URLS: tuple[str, ...] = (
    "https://upload.wikimedia.org/wikipedia/commons/8/83/Equirectangular-projection.jpg",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.200412.3x21600x10800.jpg",
)

DEFAULT_SIGMA = 0.110
DEFAULT_VERTICAL_SCALE = 0.7
DEFAULT_WIDTH = 1440
DEFAULT_HEIGHT = 720
DEFAULT_OUTPUT = "output/test.png"
DEFAULT_TOPDOWN_OUTPUT = "output/test_topdown.png"
DEFAULT_TRAJECTORY_OUTPUT = "output/test_trajectory.png"
DEFAULT_TOPDOWN_TRAJECTORY_OUTPUT = "output/test_topdown_trajectory.png"
DEFAULT_CACHE_DIR = ".cache"

DEFAULT_GD_LEARNING_RATE = 0.039
DEFAULT_GD_MAX_ITERS = 300
DEFAULT_GD_CONVERGENCE_TOL = 1e-9
DEFAULT_GD_PATIENCE = 12
DEFAULT_GD_START_OFFSET = 0.001
DEFAULT_GD_START_OFFSET_NORTH = -0.001
DEFAULT_GD_START_OFFSET_EAST = -0.002

DEFAULT_SEARCH_MODE = "ball"
DEFAULT_BALL_DT = 0.02
DEFAULT_BALL_GRAVITY = 1.0
DEFAULT_BALL_DAMPING = 5.0
DEFAULT_BALL_SPEED_TOL = 1e-4
DEFAULT_HILLY_SHARPEN_AMOUNT = 10
