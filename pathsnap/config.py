"""Central configuration for PathSnap (paths, partitions, pipeline constants).

All paths can be overridden with environment variables so the same code runs on
Laptop 1 during development and during the final evaluation (when the evaluator
testing trips are dropped into a separate folder).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data root that contains the partition folders (train/, student_test/ ...)
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", PROJECT_ROOT / "Student folder" / "Student folder"))

# Partition folder names inside DATA_ROOT
TRAIN_PARTITION = "train"
STUDENT_TEST_PARTITION = "student_test"
# Evaluator-testing trips are NOT shipped to students. At evaluation time they are
# placed in DATA_ROOT/<EVAL_PARTITION> (or PATHSNAP_EVAL_DIR points directly at them).
EVAL_PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "evaluator_test")
EVAL_DIR_OVERRIDE = os.environ.get("PATHSNAP_EVAL_DIR")

ARTIFACT_DIR = Path(os.environ.get("PATHSNAP_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MAP_DIR = REPORT_DIR / "maps"
CACHE_DIR = PROJECT_ROOT / "cache"

RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Pipeline constants (chosen from EDA on the TRAIN partition only)
# ----------------------------------------------------------------------------
# Candidate generation
CANDIDATE_RADIUS_M = 50.0      # search radius around each GPS fix
CANDIDATE_MAX_K = 12           # keep the K nearest arcs inside the radius
CANDIDATE_MIN_K = 1            # always keep at least this many (nearest fallback)
INDEX_SAMPLE_STEP_M = 10.0     # densification step for the arc spatial index
NETWORK_CROP_BUFFER_M = 1000.0 # crop the road network to the trajectory bbox + buffer

# Preprocessing
MAX_PLAUSIBLE_SPEED_MS = 70.0  # 252 km/h; faster jumps are flagged as outliers
HEADING_MIN_DISP_M = 2.0       # displacement below which the heading is unreliable
CONTEXT_WINDOW = 3             # +/- observations used for context features
