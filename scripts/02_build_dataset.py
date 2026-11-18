"""Phase 3-5: preprocessing + candidate generation + feature engineering + label alignment.

Full run (default): writes cache/features_<partition>.parquet, cache/obs_<partition>.parquet and
reports/candidate_stats_<partition>.csv for the train and student_test partitions.

Sample run (for demos):  --partitions train --limit 3
writes cache/sample_features_<partition>.parquet etc. instead, so the full cache is untouched.
The evaluator partition is never processed here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap import config, io
from pathsnap.dataset import build_partition

ap = argparse.ArgumentParser()
ap.add_argument("--partitions", default=f"{config.TRAIN_PARTITION},{config.STUDENT_TEST_PARTITION}",
                help="comma-separated: train,student_test")
ap.add_argument("--limit", type=int, default=None, help="only the first N trips per partition (sample run)")
ap.add_argument("--data-root", default=None, help="folder containing train/ and student_test/ (default: config.DATA_ROOT)")
args = ap.parse_args()
if args.data_root:
    config.DATA_ROOT = Path(args.data_root)

prefix = "sample_" if args.limit else ""
config.CACHE_DIR.mkdir(exist_ok=True)
for part in args.partitions.split(","):
    assert part in (config.TRAIN_PARTITION, config.STUDENT_TEST_PARTITION), f"partition {part} not allowed here"
    trips = io.list_trips(part)[: args.limit] if args.limit else None
    t0 = time.time()
    feats, obs, meta = build_partition(part, trips=trips, n_jobs=1 if args.limit else -1)
    feats.to_parquet(config.CACHE_DIR / f"{prefix}features_{part}.parquet", index=False)
    obs.to_parquet(config.CACHE_DIR / f"{prefix}obs_{part}.parquet", index=False)
    meta.to_csv(config.REPORT_DIR / f"{prefix}candidate_stats_{part}.csv", index=False)
    rel = obs.label_dist <= 30
    print(f"[{part}] trips={len(meta)} obs={len(obs)} candidate rows={len(feats)} "
          f"rows/obs={len(feats)/len(obs):.2f} positives={feats.y.sum()} "
          f"candidate miss rate (all)={1-obs.label_in_cands.mean():.4f} "
          f"(reliable labels)={1-obs.label_in_cands[rel].mean():.4f} reliable obs={rel.mean():.4f} "
          f"time={time.time()-t0:.1f}s")
