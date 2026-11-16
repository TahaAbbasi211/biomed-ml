"""Generate the figures used in the README from real out-of-fold predictions."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomed.datasets import BonnEEGDataset  # noqa: E402
from biomed.evaluation import out_of_fold_proba  # noqa: E402
from biomed.features import (BandPowerFeatures, CombinedFeatures,  # noqa: E402
                             StatsFeatures)
from biomed.models import MODELS  # noqa: E402

docs = ROOT / "docs"
docs.mkdir(exist_ok=True)

X, y, groups = BonnEEGDataset().load()
labels = {1: "Seizure", 2: "Tumor area", 3: "Healthy area",
          4: "Eyes closed", 5: "Eyes open"}

# 1) one example signal per class
fig, axes = plt.subplots(5, 1, figsize=(9, 8), sharex=True)
for ax, (c, name) in zip(axes, labels.items()):
    ax.plot(X[y == c][0], linewidth=0.8)
    ax.set_ylabel(name)
axes[-1].set_xlabel("sample index (178 samples = 1 second)")
fig.suptitle("Example EEG segment from each class")
fig.tight_layout()
fig.savefig(docs / "signals.png", dpi=120)
plt.close(fig)

# 2) out-of-fold predictions with the group-aware split
F = CombinedFeatures([StatsFeatures(), BandPowerFeatures()]).transform(X)
target = (y == 1).astype(int)
proba = out_of_fold_proba(MODELS["random_forest"], F, target, groups)
pred = (proba >= 0.5).astype(int)

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    target, pred, display_labels=["non-seizure", "seizure"], ax=ax,
    colorbar=False)
ax.set_title("Random forest, out-of-fold (threshold 0.5)")
fig.tight_layout()
fig.savefig(docs / "confusion_matrix.png", dpi=120)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
RocCurveDisplay.from_predictions(target, proba, ax=axes[0])
PrecisionRecallDisplay.from_predictions(target, proba, ax=axes[1])
axes[0].set_title("ROC curve")
axes[1].set_title("Precision-recall curve")
fig.tight_layout()
fig.savefig(docs / "roc_pr.png", dpi=120)
plt.close(fig)

print("confusion matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(target, pred))
print("saved:", *sorted(p.name for p in docs.glob("*.png")))
