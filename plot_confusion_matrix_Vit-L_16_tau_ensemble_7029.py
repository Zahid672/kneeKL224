import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
# Either load from CSV:
CSV_FILE = "ViT_L16_CORAL_ensemble_test_predictions_kneeKL224_TTA_tau.csv"  # Update if needed

# OR directly set arrays:
# y_true = np.array([...])
# y_pred = np.array([...])

SAVE_FIG = True
OUTPUT_FIG = "confusion_matrix_vit_l16.png"

CLASS_NAMES = ["0", "1", "2", "3", "4"]  # KL Grades


# ----------------------------------------------------
# Load predicted + ground-truth labels
# ----------------------------------------------------
df = pd.read_csv(CSV_FILE)
y_true = df["true_label"].to_numpy()
y_pred = df["predicted_label"].to_numpy()

# ----------------------------------------------------
# Compute confusion matrix
# ----------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalize per class


# ----------------------------------------------------
# Plot confusion matrix
# ----------------------------------------------------
plt.figure(figsize=(8, 7), dpi=150)
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    vmin=0,
    vmax=1,
    cbar=True
)

plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(OUTPUT_FIG, dpi=300)
    print(f"Confusion matrix saved as: {OUTPUT_FIG}")

plt.show()

# ----------------------------------------------------
# Also print raw counts for reporting
# ----------------------------------------------------
print("\nConfusion Matrix (Raw Counts):")
print(cm)
