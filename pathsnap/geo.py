"""Geodesy helpers.

Longitude/latitude are angular, so we never compute Euclidean distances in degrees.
Each trip covers at most a few tens of kilometres, so we project every trip into a
*local tangent plane* (equirectangular projection on the WGS84 ellipsoid around the
trip's reference point). Using the ellipsoid's meridional radius M and prime-vertical
radius N at the reference latitude, the distortion for a 20 km offset is < 0.1 %,
which is far below GPS noise (several metres). Haversine is provided for validation.
"""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
EARTH_R = 6371008.8


class LocalProjection:
    """Local equirectangular projection lon/lat (deg) <-> x/y (metres, east/north)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        phi = np.radians(self.lat0)
        s = np.sin(phi)
        w = np.sqrt(1 - WGS84_E2 * s * s)
        self.N = WGS84_A / w                       # prime vertical radius
        self.M = WGS84_A * (1 - WGS84_E2) / w ** 3  # meridional radius
        self.kx = np.radians(1.0) * self.N * np.cos(phi)  # metres per degree lon
        self.ky = np.radians(1.0) * self.M                # metres per degree lat

    def forward(self, lon, lat):
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def inverse(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return x / self.kx + self.lon0, y / self.ky + self.lat0

    def to_dict(self):
        return {"lon0": self.lon0, "lat0": self.lat0}


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_xy(dx, dy):
    """Compass bearing in degrees [0, 360) of a planar displacement (0 = north, 90 = east)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Absolute smallest difference between two bearings in degrees, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Vectorised distance from points P to segments AB.

    Returns (distance, t) where t in [0, 1] is the projection fraction along AB.
    All inputs broadcast against each other.
    """
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * abx + (py - ay) * aby) / l2
    t = np.where(l2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx = ax + t * abx
    qy = ay + t * aby
    return np.hypot(px - qx, py - qy), t
