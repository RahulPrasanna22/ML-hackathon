"""Phase 9: comparison plots from the ACTUAL experiment outputs in reports/experiments/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config

EXP = config.REPORT_DIR / "experiments"
FIG = config.FIG_DIR
FIG.mkdir(parents=True, exist_ok=True)
cmp = pd.read_csv(EXP / "model_comparison.csv")
folds = pd.read_csv(EXP / "cv_folds.csv")
ml = cmp[cmp.model != "nearest_road"].sort_values("cv_top1_accuracy")

# 1. CV macro-F1 and top-1 with fold std
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].barh(ml.model, ml.cv_cand_macro_f1, xerr=ml.cv_cand_macro_f1_std, color="tab:blue")
ax[0].set(title="Cross-validated candidate Macro-F1 (GroupKFold=5)", xlim=(0.8, 1.0))
ax[1].barh(ml.model, ml.cv_top1_accuracy, xerr=ml.cv_top1_accuracy_std, color="tab:green")
nr = cmp[cmp.model == "nearest_road"].cv_top1_accuracy.iloc[0]
ax[1].axvline(nr, color="k", ls="--", label=f"nearest-road baseline {nr:.3f}")
ax[1].set(title="Cross-validated observation Top-1 accuracy", xlim=(0.5, 1.0)); ax[1].legend(loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "models_cv_comparison.png", dpi=120); plt.close(fig)

# 2. Student-test top-1 and macro-F1
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(ml)); w = 0.38
ax.bar(x - w / 2, ml.st_top1_accuracy, w, label="Top-1 accuracy")
ax.bar(x + w / 2, ml.st_cand_macro_f1, w, label="Candidate Macro-F1")
ax.axhline(cmp[cmp.model == "nearest_road"].st_top1_accuracy.iloc[0], color="k", ls="--", label="nearest-road top-1")
ax.set_xticks(x, ml.model, rotation=20); ax.set(ylim=(0.5, 1.0), title="Student-test partition (10 trips)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "models_student_test.png", dpi=120); plt.close(fig)

# 3. Log loss (CV and student test), log scale because Gaussian NB is far worse
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - w / 2, ml.cv_log_loss, w, label="CV log loss")
ax.bar(x + w / 2, ml.st_log_loss, w, label="Student-test log loss")
ax.set_xticks(x, ml.model, rotation=20); ax.set(yscale="log", title="Log loss (lower = better calibrated probabilities)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "models_log_loss.png", dpi=120); plt.close(fig)

# 4. Efficiency vs accuracy
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(ml.st_inference_ms_per_obs, ml.cv_top1_accuracy, s=60 + ml.fit_all_s)
for _, r in ml.iterrows():
    ax.annotate(r.model, (r.st_inference_ms_per_obs, r.cv_top1_accuracy), textcoords="offset points", xytext=(5, 4))
ax.set(xscale="log", xlabel="inference ms per GPS fix (student test)", ylabel="CV top-1 accuracy",
       title="Accuracy vs inference cost (marker size ~ training time)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "models_efficiency.png", dpi=120); plt.close(fig)

# 5. Overfitting check: train-fold vs validation-fold top-1
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - w / 2, ml.cv_train_top1_accuracy, w, label="training folds")
ax.bar(x + w / 2, ml.cv_top1_accuracy, w, label="validation folds")
ax.set_xticks(x, ml.model, rotation=20); ax.set(ylim=(0.75, 1.0), title="Top-1 accuracy: training vs validation folds"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "models_train_vs_val.png", dpi=120); plt.close(fig)

# 6. Confusion matrices (summed over CV folds, candidate level, threshold 0.5)
models = list(ml.model)
fig, axs = plt.subplots(2, (len(models) + 1) // 2, figsize=(3.6 * ((len(models) + 1) // 2), 7))
for a, name in zip(axs.ravel(), models):
    cm = np.sum([np.array(json.loads(c)) for c in folds[folds.model == name].confusion], axis=0)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}\n{cmn[i, j]:.2%}", ha="center", va="center", fontsize=8,
                   color="white" if cmn[i, j] > .5 else "black")
    a.set(title=name, xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
for a in axs.ravel()[len(models):]:
    a.axis("off")
fig.suptitle("Candidate-level confusion matrices (sum over 5 CV folds)"); fig.tight_layout()
fig.savefig(FIG / "models_confusion_matrices.png", dpi=110); plt.close(fig)

# 7. MLP epoch history (only model with an epoch-wise training history)
h = EXP / "mlp_training_history.csv"
if h.exists():
    hd = pd.read_csv(h)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hd.epoch, hd.train_loss, "o-", label="training log loss")
    ax2 = ax.twinx(); ax2.plot(hd.epoch, hd.val_accuracy, "s-", color="tab:orange", label="internal validation accuracy")
    ax.set(xlabel="epoch", ylabel="training loss", title="MLP training history (final fit on all train rows)")
    ax2.set_ylabel("validation accuracy (10 % hold-out)")
    fig.legend(loc="center right"); fig.tight_layout(); fig.savefig(FIG / "mlp_training_history.png", dpi=120); plt.close(fig)

# 8. tuned vs untuned
tv = EXP / "tuned_vs_untuned.csv"
if tv.exists():
    t = pd.read_csv(tv)
    cols = ["cv_top1_accuracy", "cv_cand_macro_f1", "st_top1_accuracy", "st_cand_macro_f1"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xx = np.arange(len(cols))
    for k, (_, r) in enumerate(t.iterrows()):
        ax[0].bar(xx + (k - .5) * .38, r[cols].astype(float), .38, label=r.variant.split(" ")[0])
    ax[0].set_xticks(xx, cols, rotation=15); ax[0].set(ylim=(0.85, 1.0), title="Tuned vs untuned"); ax[0].legend()
    ax[1].bar(t.variant.str.split(" ").str[0], t.st_log_loss.astype(float), color=["tab:gray", "tab:green"])
    ax[1].set(title="Student-test log loss")
    fig.tight_layout(); fig.savefig(FIG / "tuned_vs_untuned.png", dpi=120); plt.close(fig)

# 9. route-level A/B/C comparison
re = EXP / "route_eval_student_test.csv"
if re.exists():
    r = pd.read_csv(re, dtype={"trip_id": str})
    meths = [("nearest_road", "A nearest road"), ("classifier_only", "B classifier only"), ("classifier_hmm", "C classifier + HMM")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    xx = np.arange(len(r)); ww = 0.27
    for k, (m, lab) in enumerate(meths):
        ax[0].bar(xx + (k - 1) * ww, r[f"{m}__accuracy"], ww, label=lab)
        ax[1].bar(xx + (k - 1) * ww, r[f"{m}__route_f1_len"], ww, label=lab)
        ax[2].bar(xx + (k - 1) * ww, r[f"{m}__norm_edit_distance"], ww, label=lab)
    for a, ttl in zip(ax, ["Pointwise accuracy", "Route F1 (length-weighted)", "Normalised edit distance (lower = better)"]):
        a.set_xticks(xx, r.trip_id.str[-3:]); a.set(title=ttl, xlabel="student-test trip")
    ax[0].set_ylim(0.3, 1.0); ax[1].set_ylim(0.4, 1.0); ax[2].legend(fontsize=8, loc="upper center")
    fig.tight_layout(); fig.savefig(FIG / "route_eval_abc.png", dpi=120); plt.close(fig)
print("plots written to", FIG)
