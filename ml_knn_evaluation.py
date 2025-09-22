# ml_knn_evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ---------- Setup ----------
os.makedirs("plots", exist_ok=True)  # for saving figures

# ---------- Load data ----------
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ---------- Train kNN(k=5) ----------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ---------- Predictions ----------
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
plt.title("kNN (k=5) — Confusion Matrix")
plt.tight_layout()
fig_cm.savefig("plots/confusion_matrix_knn5.png", dpi=200)
plt.close(fig_cm)

# ---------- Classification Report ----------
report = classification_report(y_test, y_pred, target_names=class_names)
print("\nClassification Report:\n", report)

# Save report to a text file (optional)
with open("plots/classification_report.txt", "w") as f:
    f.write(report)

# ---------- ROC & AUC (one-vs-rest for 3 classes) ----------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # shape: (n_samples, 3)
n_classes = y_test_bin.shape[1]

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
fig_roc, ax = plt.subplots(figsize=(6, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, color in zip(range(n_classes), colors):
    ax.plot(
        fpr[i], tpr[i], color=color, lw=2,
        label=f"ROC — {class_names[i]} (AUC = {roc_auc[i]:.2f})"
    )
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — kNN (k=5)")
ax.legend(loc="lower right")
plt.tight_layout()
fig_roc.savefig("plots/roc_knn5.png", dpi=200)
plt.close(fig_roc)

# ---------- Small summary to console ----------
acc = (y_pred == y_test).mean()
print(f"\nAccuracy: {acc:.4f}")
