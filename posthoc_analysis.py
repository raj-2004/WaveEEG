# ============================================================================
# PURE POST-HOC ANALYSIS SCRIPT (NO MODEL, NO DATASET)
# ============================================================================
# Uses ONLY saved outputs from training/evaluation
# Reviewer-safe
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import cdist

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = "final_results"
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
OUT_DIR = os.path.join(BASE_DIR, "posthoc_analysis")

FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Load saved results ONLY
# ----------------------------------------------------------------------------
print("Loading saved evaluation results...")

X = np.load(os.path.join(METRICS_DIR, "tsne_embeddings.npy"))
y_true = np.load(os.path.join(METRICS_DIR, "y_true.npy"))
y_pred = np.load(os.path.join(METRICS_DIR, "y_pred.npy"))
y_probs = np.load(os.path.join(METRICS_DIR, "y_probs.npy"))

num_classes = y_probs.shape[1]
class_names = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]

confidence = np.max(y_probs, axis=1)
correct = y_true == y_pred

# ----------------------------------------------------------------------------
# t-SNE projection (from saved embeddings)
# ----------------------------------------------------------------------------
print("Running t-SNE on saved embeddings...")
tsne = TSNE(n_components=2, random_state=42, max_iter=1500)
X_2d = tsne.fit_transform(X)

# ----------------------------------------------------------------------------
# FIGURE 1: t-SNE (true classes)
# ----------------------------------------------------------------------------
plt.figure(figsize=(8, 7))
sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="tab10", s=30)
plt.colorbar(sc, ticks=range(num_classes))
plt.title("t-SNE: True Classes")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_tsne_true_classes.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# FIGURE 2: t-SNE (confidence)
# ----------------------------------------------------------------------------
plt.figure(figsize=(8, 7))
sc = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=confidence, cmap="viridis", s=30)
plt.colorbar(sc, label="Confidence")
plt.title("t-SNE: Prediction Confidence")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_tsne_confidence.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# FIGURE 3: Correct vs Incorrect
# ----------------------------------------------------------------------------
plt.figure(figsize=(8, 7))
plt.scatter(X_2d[correct, 0], X_2d[correct, 1], s=25, label="Correct", alpha=0.6)
plt.scatter(X_2d[~correct, 0], X_2d[~correct, 1], s=25, label="Incorrect", alpha=0.6)
plt.legend()
plt.title("t-SNE: Correct vs Incorrect")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_tsne_correct_vs_incorrect.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# FIGURE 4: Confusion matrix (row-normalized)
# ----------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_confusion_matrix_normalized.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# FIGURE 5: ROC curves
# ----------------------------------------------------------------------------
y_bin = label_binarize(y_true, classes=range(num_classes))

plt.figure(figsize=(8, 7))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=class_names[i])

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_roc_curves.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# FIGURE 6: Precision–Recall curves
# ----------------------------------------------------------------------------
plt.figure(figsize=(8, 7))
for i in range(num_classes):
    p, r, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
    plt.plot(r, p, label=class_names[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_pr_curves.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# FIGURE 7: Calibration curve
# ----------------------------------------------------------------------------
prob_true, prob_pred = calibration_curve(correct.astype(int), confidence, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k--")
plt.plot(prob_pred, prob_true, "o-")
plt.xlabel("Predicted Confidence")
plt.ylabel("Empirical Accuracy")
plt.title("Calibration Curve")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_calibration_curve.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# TABLE: Per-class metrics
# ----------------------------------------------------------------------------
report = classification_report(
    y_true, y_pred, target_names=class_names,
    output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    os.path.join(TAB_DIR, "per_class_metrics.csv")
)

print("✅ PURE POST-HOC ANALYSIS COMPLETE")
print(f"All outputs saved to: {OUT_DIR}")
