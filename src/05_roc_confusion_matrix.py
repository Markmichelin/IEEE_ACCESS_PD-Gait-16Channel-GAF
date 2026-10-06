import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# =========================
# Read data
# =========================
swin_df = pd.read_csv(
    "independent_test_predictions_subject_swin_tiny_patch4_window7_224.csv"
)

conv_df = pd.read_csv(
    "independent_test_predictions_subject_convnext_tiny.csv"
)

# =========================
# Swin-Tiny
# =========================
swin_true = swin_df["y_true"]
swin_prob = swin_df["prob_PD"]

swin_fpr, swin_tpr, _ = roc_curve(swin_true, swin_prob)
swin_auc = roc_auc_score(swin_true, swin_prob)

# =========================
# ConvNeXt-Tiny
# =========================
conv_true = conv_df["y_true"]
conv_prob = conv_df["prob_PD"]

conv_fpr, conv_tpr, _ = roc_curve(conv_true, conv_prob)
conv_auc = roc_auc_score(conv_true, conv_prob)

# =========================
# Combined ROC
# =========================
plt.figure(figsize=(7, 7))

plt.plot(
    swin_fpr,
    swin_tpr,
    linewidth=2.5,
    label=f"Swin-Tiny 1024×1024 (AUC = {swin_auc:.3f})"
)

plt.plot(
    conv_fpr,
    conv_tpr,
    linewidth=2.5,
    label=f"ConvNeXt-Tiny 512×512 (AUC = {conv_auc:.3f})"
)

# Chance line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Chance"
)

plt.xlabel("1 − Specificity", fontsize=13)
plt.ylabel("Sensitivity", fontsize=13)

plt.xlim(0, 1)
plt.ylim(0, 1.02)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.legend(
    loc="lower right",
    frameon=False,
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    "ROC_Combined_Swin1024_ConvNeXt512.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "ROC_Combined_Swin1024_ConvNeXt512.pdf",
    bbox_inches="tight"
)

plt.show()