from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Model discrimination summary
metrics = pd.read_csv(REPORTS / "model_results.csv").set_index("Model")[["ROC-AUC", "PR-AUC"]]
ax = metrics.plot(kind="bar", figsize=(8, 5))
ax.set_ylabel("Score")
ax.set_title("Model Discrimination Summary")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=0)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIGURES / "01_model_discrimination_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Confusion matrices
cm = pd.read_csv(REPORTS / "confusion_matrices.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (_, row) in zip(axes, cm.iterrows()):
    matrix = [[row["True Negative"], row["False Positive"]],
              [row["False Negative"], row["True Positive"]]]
    sns.heatmap(
        matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Predicted No Default", "Predicted Default"],
        yticklabels=["Actual No Default", "Actual Default"]
    )
    ax.set_title(row["Model"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
fig.suptitle("Confusion Matrices at the 0.5 Threshold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES / "02_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. Random Forest feature importance
fi = pd.read_csv(REPORTS / "feature_importance.csv").sort_values("importance", ascending=True)
plt.figure(figsize=(9, 6))
plt.barh(fi["feature"], fi["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance (Top 10)")
plt.tight_layout()
plt.savefig(FIGURES / "03_random_forest_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

# 4. Feature-group contribution
fg = pd.read_csv(REPORTS / "feature_group_results.csv")
plt.figure(figsize=(9, 5))
plt.plot(fg["Feature Set"], fg["ROC-AUC"], marker="o", linewidth=2)
plt.ylabel("ROC-AUC")
plt.xlabel("Feature Set")
plt.title("Predictive Performance with Incremental Feature Groups")
plt.ylim(0.55, 0.80)
plt.tight_layout()
plt.savefig(FIGURES / "04_feature_group_contribution.png", dpi=300, bbox_inches="tight")
plt.close()

# 5. Customer risk profile
rp = pd.read_csv(REPORTS / "customer_risk_profiles.csv")
plot = rp[rp["Profile"].isin(["Low Risk", "Medium Risk", "High Risk"])].copy()
plt.figure(figsize=(8, 5))
plt.bar(plot["Profile"], plot["Observed Default Rate"])
plt.ylabel("Observed Default Rate (%)")
plt.xlabel("Repayment Risk Profile")
plt.title("Observed Default Rate by Customer Risk Profile")
for i, value in enumerate(plot["Observed Default Rate"]):
    plt.text(i, value + 2, f"{value:.1f}%", ha="center")
plt.ylim(0, plot["Observed Default Rate"].max() + 12)
plt.tight_layout()
plt.savefig(FIGURES / "05_customer_risk_profile_default_rate.png", dpi=300, bbox_inches="tight")
plt.close()

print("Portfolio figures generated in", FIGURES)
