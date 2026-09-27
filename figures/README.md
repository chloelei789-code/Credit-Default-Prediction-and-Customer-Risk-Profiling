# Figures

These PNG files are selected portfolio assets for GitHub display. The notebook also contains code to generate the model-specific ROC and Precision-Recall curves after the raw dataset is available locally.

- `01_model_discrimination_summary.png`: exact recorded ROC-AUC and PR-AUC values for the two models.
- `02_confusion_matrices.png`: exact recorded confusion matrices at the 0.5 threshold.
- `03_random_forest_feature_importance.png`: top 10 impurity-based feature importance values from the recorded Random Forest run.
- `04_feature_group_contribution.png`: exact recorded ROC-AUC by feature group.
- `05_customer_risk_profile_default_rate.png`: observed default rates for the Low, Medium, and High Risk descriptive profiles.

Run `python src/generate_portfolio_figures.py` to regenerate these summary figures from the recorded result tables in `reports/`.
