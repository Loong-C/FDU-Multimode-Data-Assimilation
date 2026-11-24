"""Nominal BWF court geometry in metres, centred on the floor beneath the net.

x is longitudinal, y transverse and z vertical. The line segments locate the
nominal boundaries from Laws of Badminton, Section 4.1, Diagram A. Painted
lines have finite width; the segments are references for plotting or projection.
"""

import numpy as np


HALF_LENGTH = 6.70
SINGLES_HALF_WIDTH = 2.59
DOUBLES_HALF_WIDTH = 3.05
SHORT_SERVICE_X = 1.98
DOUBLES_LONG_SERVICE_X = HALF_LENGTH - 0.76
LINE_WIDTH = 0.04
NET_CENTER_HEIGHT = 1.524
NET_POST_HEIGHT = 1.55
NET_DEPTH = 0.76


def floor_lines():
    """Return named pairs of (x, y, z) endpoints including both service boxes."""
    lines = {}
    for sign, side in ((-1, "negative"), (1, "positive")):
        for name, distance in (
            ("baseline", HALF_LENGTH),
            ("short_service", SHORT_SERVICE_X),
            ("doubles_long_service", DOUBLES_LONG_SERVICE_X),
        ):
            x = sign * distance
            lines[f"{name}_{side}_x"] = ((x, -DOUBLES_HALF_WIDTH, 0), (x, DOUBLES_HALF_WIDTH, 0))
        for name, width in (("singles", SINGLES_HALF_WIDTH), ("doubles", DOUBLES_HALF_WIDTH)):
            y = sign * width
            lines[f"{name}_sideline_{side}_y"] = ((-HALF_LENGTH, y, 0), (HALF_LENGTH, y, 0))
        lines[f"centre_service_{side}_x"] = ((sign * SHORT_SERVICE_X, 0, 0), (sign * HALF_LENGTH, 0, 0))
    return lines


def net_height(y):
    """Top of net at x=0; quadratic sag between regulation heights, NaN beyond posts."""
    y = np.asarray(y, dtype=float)
    height = NET_CENTER_HEIGHT + (NET_POST_HEIGHT - NET_CENTER_HEIGHT) * (y / DOUBLES_HALF_WIDTH) ** 2
    result = np.where(np.abs(y) <= DOUBLES_HALF_WIDTH, height, np.nan)
    return result.item() if result.ndim == 0 else result
