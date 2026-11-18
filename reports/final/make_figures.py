"""Collect / generate the figures used by the final report (reports/final/figures/).

Reuses existing project figures and screenshots; generates only:
  * architecture diagram (drawn, no data)
  * evaluator-test per-trip accuracy chart   (from reports/evaluator/evaluator_results.csv)
  * evaluator-test route-quality chart        (same source)
  * cropped dashboard screenshots              (from reports/integration/*.png)
  * a Folium route-reconstruction map screenshot (student-test trip, rendered with headless Edge)
Run with the project venv:  .venv\\Scripts\\python reports\\final\\make_figures.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. reuse existing figures
COPY = {
    "eda_trip_locations.png": "reports/figures/eda_trip_locations.png",
    "eda_trip_sizes.png": "reports/figures/eda_trip_sizes.png",
    "eda_sampling_speed.png": "reports/figures/eda_sampling_speed.png",
    "eda_label_distance_rank.png": "reports/figures/eda_label_distance_rank.png",
    "eda_trajectories.png": "reports/figures/eda_trajectories.png",
    "eda_network_connectivity.png": "reports/figures/eda_network_connectivity.png",
    "preprocessing_00000003.png": "reports/preprocessing_demo/preprocessing_00000003.png",
    "models_cv_comparison.png": "reports/figures/models_cv_comparison.png",
    "models_log_loss.png": "reports/figures/models_log_loss.png",
    "models_train_vs_val.png": "reports/figures/models_train_vs_val.png",
    "models_efficiency.png": "reports/figures/models_efficiency.png",
    "models_confusion_matrices.png": "reports/figures/models_confusion_matrices.png",
    "mlp_training_history.png": "reports/figures/mlp_training_history.png",
    "optuna_xgboost_history.png": "reports/figures/optuna_xgboost_history.png",
    "optuna_xgboost_importance.png": "reports/figures/optuna_xgboost_importance.png",
    "tuned_vs_untuned.png": "reports/figures/tuned_vs_untuned.png",
    "route_eval_abc_student_test.png": "reports/figures/route_eval_abc.png",
}
for dst, src in COPY.items():
    shutil.copy(ROOT / src, OUT / dst)


# ---------------------------------------------------------------- 2. architecture diagram
def box(ax, x, y, w, h, text, fc, ec="#1b2a44", fs=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#34495e"))
    if text:
        ax.text((x1 + x2) / 2 + 0.08, (y1 + y2) / 2, text, fontsize=8, color="#34495e", va="center")


fig, ax = plt.subplots(figsize=(11, 8.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 10); ax.axis("off")
# laptop frames
ax.add_patch(FancyBboxPatch((0.2, 7.6), 4.0, 2.2, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#fdf6e3", ec="#b58900", lw=1.5, ls="--"))
ax.text(0.35, 9.55, "Laptop 2: evaluator UI", fontsize=10, weight="bold", color="#8a6d00")
box(ax, 0.5, 7.85, 3.4, 1.4, "React 19 + TypeScript + Vite\nLeaflet map · metrics · charts\nmodel comparison · history", "#fff3c4")
ax.add_patch(FancyBboxPatch((4.9, 0.2), 5.9, 9.6, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#eef5fb", ec="#2b6cb0", lw=1.5, ls="--"))
ax.text(5.05, 9.55, "Laptop 1: model + API (Python)", fontsize=10, weight="bold", color="#1f4e85")
steps = [
    ("FastAPI backend  (api/main.py)\n/health · /evaluation/trips · /evaluation/run · /models/comparison", "#d6e9fb"),
    ("GPS preprocessing  (pathsnap/preprocess.py)\nclean · sort · project to metres · speed/heading", "#e3f1e6"),
    ("Candidate generation  (pathsnap/network.py, features.py)\nKD-tree, ≤ 12 arcs within 50 m", "#e3f1e6"),
    ("Feature extraction: 28 features per (fix, arc)", "#e3f1e6"),
    ("Tuned XGBoost classifier  (artifacts/final/model.joblib)\nP(candidate is the driven road)", "#fde2e2"),
    ("HMM / Viterbi decoder  (pathsnap/hmm.py)\nemissions + network-distance transitions", "#fde2e2"),
    ("Route reconstruction: shortest-path gap filling", "#ece3f7"),
    ("Server-side scoring (pathsnap/evaluation.py)\n.route read only AFTER prediction", "#f3e6d6"),
]
y = 8.6
h = 0.74
for i, (t, c) in enumerate(steps):
    box(ax, 5.2, y - h / 2, 5.3, h, t, c, fs=8.6, weight="bold" if i == 0 else "normal")
    if i:
        arrow(ax, 7.85, y + h / 2 + 0.31, 7.85, y + h / 2 + 0.02)
    y -= 1.07
arrow(ax, 3.9, 8.75, 5.2, 8.75, None)
ax.text(4.02, 8.95, "HTTP/JSON\ntrip_id", fontsize=8, color="#34495e")
ax.annotate("", xy=(3.9, 8.25), xytext=(5.2, 8.25), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#34495e"))
ax.text(3.95, 7.6, "predicted route\n[lon,lat] + metrics\n(no ground truth)", fontsize=8, color="#34495e")
box(ax, 0.5, 4.6, 3.4, 1.5, "Trip data (per trip)\n.track  .nodes  .arcs\n(+ .route for scoring only)", "#f4f4f4")
arrow(ax, 3.9, 5.35, 5.2, 6.2, None)
box(ax, 0.5, 2.0, 3.4, 1.7, "Offline development (TRAIN only)\n8 models · GroupKFold(5)\nOptuna · HMM tuning on OOF\n→ frozen artifacts/final/", "#f4f4f4")
arrow(ax, 3.9, 2.85, 5.2, 4.75, None)
ax.set_title("Figure: PathSnap end-to-end architecture (two-laptop deployment)", fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(OUT / "architecture.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 3. evaluator-test charts (official partition)
ev = pd.read_csv(ROOT / "reports" / "evaluator" / "evaluator_results.csv", dtype={"trip_id": str}).sort_values("trip_id")
x = np.arange(len(ev)); w = 0.27
fig, ax = plt.subplots(figsize=(11, 4.3))
for k, (col, lab, colr) in enumerate([("nearest_top1_accuracy", "A: nearest road", "#9aa5b1"),
                                      ("clf_top1_accuracy", "B: XGBoost only", "#f0a03c"),
                                      ("hmm_top1_accuracy", "C: XGBoost + HMM/Viterbi", "#1f9bb0")]):
    ax.bar(x + (k - 1) * w, ev[col], w, label=lab, color=colr)
ax.set_xticks(x, ev.trip_id); ax.set_ylim(0.3, 1.13); ax.set_ylabel("pointwise top-1 accuracy")
ax.set_title("Official evaluator-test partition: per-trip accuracy (frozen model, each trip evaluated once)")
ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "evaluator_accuracy_per_trip.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for a, (metric, title) in zip(axs, [("route_f1", "Route F1 (length-weighted)"), ("sequence_accuracy", "Sequence accuracy"),
                                     ("network_reachable_continuity", "Network-reachable continuity")]):
    a.bar(x - 0.2, ev[f"clf_{metric}"], 0.4, label="XGBoost only", color="#f0a03c")
    a.bar(x + 0.2, ev[f"hmm_{metric}"], 0.4, label="XGBoost + HMM", color="#1f9bb0")
    a.set_xticks(x, ev.trip_id.str[-3:], fontsize=8); a.set_title(title, fontsize=10); a.set_ylim(0.4, 1.02); a.grid(axis="y", alpha=.3)
axs[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Official evaluator-test partition: effect of HMM/Viterbi on route quality", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "evaluator_route_quality.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 4. cropped dashboard screenshots
def crop(src, dst, box_frac):
    im = Image.open(ROOT / src)
    W, H = im.size
    l, t, r, b = box_frac
    im.crop((int(l * W), int(t * H), int(r * W), int(b * H))).save(OUT / dst, optimize=True)


crop("reports/integration/ui_trip_00000015.png", "dashboard_evaluator_00000015.png", (0, 0, 1, 0.53))
crop("reports/integration/ui_trip_00000015.png", "dashboard_hmm_chart_00000015.png", (0.25, 0.82, 1, 0.97))
crop("reports/integration/ui_model_comparison_page.png", "dashboard_model_comparison.png", (0.25, 0.415, 1, 0.97))

# ---------------------------------------------------------------- 5. Folium map screenshot (student-test, offline map)
from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1400, "height": 850})
    pg.goto((ROOT / "reports" / "maps" / "match_student_test_00000009.html").as_uri())
    pg.wait_for_timeout(2500)
    pg.screenshot(path=str(OUT / "folium_route_student_test_00000009.png"))
    pg.goto((ROOT / "reports" / "preprocessing_demo" / "map_00000003.html").as_uri())
    pg.wait_for_timeout(2500)
    pg.screenshot(path=str(OUT / "folium_preprocessing_00000003.png"))
    b.close()

print("figures:", sorted(p.name for p in OUT.iterdir()))
