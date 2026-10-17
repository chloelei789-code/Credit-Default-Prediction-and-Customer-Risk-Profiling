# Credit Default Prediction and Customer Risk Profiling

> **1-minute portfolio view:** a beginner-friendly, end-to-end data science project on a public credit-card dataset. It combines data understanding, exploratory analysis, feature engineering, Logistic Regression, Random Forest, model evaluation beyond accuracy, feature-group contribution, and an interpretable rule-based customer risk profile.

## 1. Project Background and Objective

Credit-risk analysis is a practical setting in which data quality, customer behavior, model performance, and decision trade-offs all matter. This project asks:

**Can historical customer characteristics and repayment behavior help identify next-month default, and can those patterns be translated into an understandable customer risk profile?**

The goal is an interpretable, reproducible analysis rather than a production credit-scoring model. Results are associations observed in one public dataset and should not be read as causal or deployment-ready conclusions.

## 2. Dataset

- **Source:** UCI Machine Learning Repository, *Default of Credit Card Clients* (Yeh, 2009). See [`data/README.md`](data/README.md) for the source link and citation.
- **Local file:** `dataset/default_of_credit_card_clients_dataset.csv` (30,000 customers, 23 original variables plus the target).
- **Target:** `default.payment.next.month` (1 = default, 0 = no default). The default rate is 22.12%.
- **Period:** six months of repayment status, bill amounts, and payment amounts per customer.

Negative bill amounts are kept exactly as reported; they are never clipped to zero. The payment-to-bill ratio is defined only when the six-month bill total is positive. Otherwise the ratio is left missing and flagged, and those customers are handled explicitly in modeling and profiling.

## 3. Methods Overview

### Data preparation

The analysis checks shape, data types, missing values, duplicates, and target balance. It inspects negative bills without changing them, confirms that only the payment-to-bill ratio has systematic missing values (1,071 customers with a non-positive six-month bill total), and verifies that no other modeling field or target is missing.

### Feature engineering

Six-month summaries are built from the monthly columns:

| Feature | Meaning |
|---|---|
| `delay_months` | number of months with a positive repayment delay |
| `avg_bill_amt`, `bill_amt_std` | six-month bill level and volatility |
| `avg_payment`, `payment_std` | six-month payment level and volatility |
| `payment_to_bill_ratio_capped` | cumulative payment-to-bill ratio, capped at 5 |
| `ratio_missing` | flag for customers whose ratio cannot be computed |

Twelve features enter the model: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, and the engineered features above. `SEX`, `EDUCATION`, and `MARRIAGE` are treated as categorical variables; numeric features are median-imputed and standardized, and categorical features are one-hot encoded. Preprocessing is fitted on the training set only.

### Models and evaluation

- **Logistic Regression** — an interpretable linear baseline.
- **Random Forest** — a nonlinear comparison model.

Both models use `class_weight="balanced"` and a fixed `random_state=42`. The train/test split is stratified 80/20 (24,000 / 6,000 customers, default rate 22.12% in both). Evaluation uses Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC (implemented with `average_precision_score`, i.e. Average Precision), together with confusion matrices, ROC and PR curves, and a threshold table. ROC-AUC measures ranking ability and is not prediction accuracy.

## 4. Key Results

### Model performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7535 | 0.4541 | 0.5659 | **0.5039** | **0.7501** | 0.4901 |
| Random Forest | 0.8007 | **0.5956** | 0.3075 | 0.4056 | 0.7493 | **0.4933** |

The reference rule "predict no default for everyone" reaches 77.88% accuracy with zero default recall, so Accuracy alone cannot evaluate a credit-default model. The two models rank customers almost equally well, but they behave differently at the 0.5 threshold: Logistic Regression finds more defaulters (751 identified, 576 missed, 903 false positives), while Random Forest produces fewer false positives (277) but misses more defaulters (919). The threshold table shows this trade-off directly: moving the Logistic Regression threshold from 0.5 to 0.3 raises Recall from 56.59% to 93.97% while Precision falls from 45.41% to 25.70%. The table only demonstrates the trade-off and does not select a best threshold from the test set.

### Random Forest feature importance

The top impurity-based importances are `delay_months` (0.1539), `bill_amt_std` (0.1186), and `payment_to_bill_ratio_capped` (0.1169), followed by `avg_payment` (0.1153) and `avg_bill_amt` (0.1131). These values describe how much each feature contributes to tree splits; they do not indicate the direction of an effect or how much prediction accuracy would change.

### Feature-group contribution

Adding behavioral groups cumulatively to profile and credit variables gives:

| Feature set | ROC-AUC | F1 | Delta ROC-AUC |
|---|---:|---:|---:|
| Demographic + Credit | 0.6027 | 0.3823 | — |
| + Delay Behavior | 0.7296 | 0.4879 | **+0.1269** |
| + Payment-to-Bill Behavior | 0.7409 | 0.5009 | +0.0113 |
| + Billing Behavior | 0.7411 | 0.5012 | +0.0003 |
| + Payment Behavior | 0.7501 | 0.5039 | +0.0090 |

Delay behavior provides the largest single improvement. The payment-to-bill ratio adds a smaller but clearly positive increment beyond the delay count, billing adds almost nothing once these variables are included, and payment amounts add a modest final gain. The comparison is exploratory: the group order is chosen by the author, and another model or feature order could change the intermediate values.

### Customer risk profiling

The descriptive profile combines `delay_months` with `payment_to_bill_ratio_capped`:

| Profile | Customers | Observed default rate |
|---|---:|---:|
| Low Risk | 13,562 | **11.24%** |
| Medium Risk | 11,848 | **21.94%** |
| High Risk | 3,519 | **60.93%** |
| Unknown | 1,071 | **34.36%** |

The rules are descriptive, not model predictions. **Low Risk** means 0 delayed months and a ratio above the lowest tercile; **High Risk** means 3-6 delayed months and a ratio below the highest tercile; other customers with a computable ratio form **Medium Risk**. The delay tier dominates the split, and the ratio mainly separates customers who already have delay records. **Unknown** contains the 1,071 customers (3.57%) whose six-month bill total is not positive; they should not be treated as a fourth risk tier.

### Selected figures

![Recorded model discrimination summary](figures/09_model_discrimination_summary.png)

![Recorded confusion matrices at the 0.5 threshold](figures/10_confusion_matrices.png)

![Top 10 Random Forest feature importances](figures/11_random_forest_feature_importance.png)

![Recorded ROC-AUC by cumulative feature group](figures/12_feature_group_contribution.png)

![Observed default rate by customer risk profile](figures/13_customer_risk_profile_default_rate.png)

## 5. Main Conclusions

1. **Delay behavior is the strongest incremental information source.** Adding `delay_months` raises ROC-AUC from 0.6027 to 0.7296 (+0.1269), the largest single step in the feature-group analysis.
2. **The payment-to-bill ratio adds information beyond delay counts.** It raises ROC-AUC from 0.7296 to 0.7409 (+0.0113), which is why the project does not rely on `delay_months` alone.
3. **The rule-based profile separates clearly different observed default rates.** The High Risk group defaults at about 60.9%, compared with about 11.2% in the Low Risk group.
4. **Error trade-offs matter more than a single accuracy number.** Logistic Regression captures more defaulters at the 0.5 threshold, Random Forest raises fewer false alerts, and the preferred operating point depends on the relative cost of missed defaults and false alerts.

## 6. Limitations

- The analysis uses a single public dataset, so results may not generalize to another bank, country, or customer population.
- The data are observational; associations should not be interpreted as causal effects.
- Engineered variables such as `delay_months` and `payment_to_bill_ratio_capped` are simplified behavioral proxies.
- The features, feature groups, and profiling rules were chosen after exploratory analysis of the full sample, and model evaluation uses one stratified 80/20 hold-out split with `random_state=42`; test-set metrics are therefore exploratory rather than fully independent confirmation.
- The model scores are not probability-calibrated and are not evaluated for fairness or subgroup performance.
- Random Forest impurity importance is descriptive and can be affected by correlated features.

## 7. Reproduction

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add the dataset

Download the UCI dataset described in [`data/README.md`](data/README.md) and place the CSV at:

```text
dataset/default_of_credit_card_clients_dataset.csv
```

The notebook reads this path relative to the `notebooks/` folder.

### Run the analysis

Open `notebooks/Credit_Default_Project2.ipynb` and run the cells from top to bottom, or execute a full headless run:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Credit_Default_Project2.ipynb --ExecutePreprocessor.timeout=900
```

A full run regenerates the model figures in `figures/` and rewrites the result tables in `reports/`, keeping the recorded numbers in sync with the code.

### Regenerate the README summary figures

The five committed summary figures are rebuilt from the recorded result tables:

```bash
python src/generate_portfolio_figures.py
```

### View the HTML report

An exported version of the executed notebook is available at [`reports/Credit_Default_Project.html`](reports/Credit_Default_Project.html).

## 8. Repository Structure

```text
Credit-Default-Prediction-and-Customer-Risk-Profiling/
├── README.md
├── AGENTS.md                                  # contributor guide
├── requirements.txt
├── data/
│   └── README.md                              # source, citation, and dataset notes
├── dataset/
│   └── default_of_credit_card_clients_dataset.csv
├── figures/
│   ├── README.md
│   ├── 01-08_*.png                            # figures written by the notebook
│   └── 09-13_*.png                            # summary figures built from reports/
├── models/
│   └── README.md                              # optional local model artifacts
├── notebooks/
│   └── Credit_Default_Project2.ipynb          # the project notebook
├── reports/
│   ├── README.md
│   ├── Credit_Default_Project.html
│   └── *.csv                                  # rewritten by the notebook on every full run
└── src/
    ├── README.md
    └── generate_portfolio_figures.py
```

## 9. Academic Positioning

This is a self-directed portfolio project that connects financial-domain experience with practical data science. It demonstrates an end-to-end workflow: **data understanding -> feature construction -> modeling -> evaluation -> interpretation -> communication**. It is intentionally scoped to explainable methods and does not present itself as a production credit-scoring solution.

> All documented results are associations observed within the public dataset. They are not causal claims and do not show that the models are ready for real-world lending decisions.
