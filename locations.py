"""Default destination locations for world hill generation."""

DEFAULT_LOCATIONS: list[tuple[float, float]] = [
    (4.90, 52.37),       # Amsterdam
    (-84.09, 9.93),      # San Jose (capital of Costa Rica)
    (72.88, 19.08),      # Mumbai
    (33.43, 35.13),      # Cyprus (island center)
    (172.64, -43.53),    # Christchurch
    (-74.00, 40.71),     # New York
    (100.50, 13.76),     # Bangkok (capital of Thailand)
    (47.51, -18.88),     # Antananarivo (capital of Madagascar)
    (-77.04, -12.05),    # Lima
]

# (longitude, latitude) in degrees WGS84 — approximate centers / trailheads.
DESTINATIONS: list[tuple[float, float]] = [
    (-103.25, 29.25),       # Big Bend National Park, Texas
    (32.32, 35.05),        # Akamas Peninsula, Cyprus
    (125.96, 9.87),        # Siargao Island (General Luna area), Philippines
    (104.98, 22.82),       # Ha Giang, Vietnam
    (6.752, 60.124),       # Trolltunga trailhead area, Norway
    (98.99, 18.79),        # Chiang Mai, Thailand
    (169.357, -44.562),    # Breast Hill, Otago (LINZ-style coords)
    (81.312, 30.675),      # Mount Kailash, Tibet / China
    (-71.30, -13.87),      # Rainbow Mountain (Vinicunca), Peru
]
