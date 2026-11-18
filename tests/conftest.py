"""Shared fixtures. Synthetic data here is ONLY for isolated unit tests.

EVALUATOR GUARD: the test suite must never read the evaluator partition (trips or .route
labels), even if the shell that runs pytest has PATHSNAP_EVAL_DIR / PATHSNAP_EVAL_PARTITION
set for the API. Before any pathsnap module is imported we:
  1. remove those variables from the environment (also for subprocesses spawned by tests),
  2. point the evaluator partition name at a non-existent placeholder,
  3. wrap pathsnap.io.partition_dir so that ANY path containing an "evaluator_test"
     component, or the original PATHSNAP_EVAL_DIR, raises instead of being read.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_BLOCKED_EVAL_DIR = os.environ.pop("PATHSNAP_EVAL_DIR", None)
os.environ.pop("PATHSNAP_EVAL_PARTITION", None)
EVAL_BLOCK_NAMES = {"evaluator_test"}

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap import config as _config, io as _io  # noqa: E402
from pathsnap.geo import LocalProjection  # noqa: E402

_config.EVAL_DIR_OVERRIDE = None
_config.EVAL_PARTITION = "__evaluator_partition_blocked_during_tests__"
_original_partition_dir = _io.partition_dir


class EvaluatorAccessError(RuntimeError):
    pass


def _guarded_partition_dir(partition: str) -> Path:
    d = _original_partition_dir(partition)
    parts = {p.lower() for p in Path(d).parts}
    blocked = parts & EVAL_BLOCK_NAMES or partition.lower() in EVAL_BLOCK_NAMES
    if _BLOCKED_EVAL_DIR:
        try:
            blocked = blocked or Path(d).resolve() == Path(_BLOCKED_EVAL_DIR).resolve()
        except OSError:
            pass
    if blocked:
        raise EvaluatorAccessError(f"test suite tried to access the evaluator partition: {d}")
    return d


_io.partition_dir = _guarded_partition_dir

LON0, LAT0 = 8.0, 50.0


@pytest.fixture
def proj():
    return LocalProjection(LON0, LAT0)


def grid_network(n: int = 5, spacing: float = 100.0, two_way: bool = True):
    """n x n grid of nodes `spacing` metres apart; arcs between 4-neighbours."""
    p = LocalProjection(LON0, LAT0)
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    lon, lat = p.inverse(xs.ravel(), ys.ravel())
    nodes = np.c_[lon, lat]
    arcs = []
    idx = lambda r, c: r * n + c
    for r in range(n):
        for c in range(n):
            for dr, dc in [(0, 1), (1, 0)]:
                rr, cc = r + dr, c + dc
                if rr < n and cc < n:
                    arcs.append((idx(r, c), idx(rr, cc)))
                    if two_way:
                        arcs.append((idx(rr, cc), idx(r, c)))
    return nodes, np.array(arcs, dtype=np.int64)


@pytest.fixture
def grid():
    return grid_network()


def straight_track(y_m: float = 3.0, x0: float = 5.0, x1: float = 395.0, speed: float = 10.0, noise: float = 0.0, seed: int = 0):
    """Vehicle driving east along the bottom row (y ~ 0) of the grid."""
    p = LocalProjection(LON0, LAT0)
    rng = np.random.default_rng(seed)
    x = np.arange(x0, x1, speed)
    y = np.full_like(x, y_m) + rng.normal(0, noise, len(x))
    lon, lat = p.inverse(x, y)
    return pd.DataFrame({"lon": lon, "lat": lat, "t": np.arange(len(x), dtype=float)})
