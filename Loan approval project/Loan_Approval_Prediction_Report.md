# Loan Approval Prediction — Machine Learning Project
### Dataset: `loan_approval_dataset.csv`

## 1. Problem Statement

Build a binary classifier that predicts whether a loan application will be
**Approved** or **Rejected**, based on applicant financials and credit
profile.

## 2. Dataset

**4,269 applications, 13 columns.** Notably larger and richer in financial
detail than a typical toy loan dataset — it includes actual asset values
rather than just income.

| Column | Description |
|---|---|
| loan_id | Unique ID (dropped) |
| no_of_dependents | Number of dependents |
| education | Graduate / Not Graduate |
| self_employed | Yes / No |
| income_annum | Annual income |
| loan_amount | Loan amount requested |
| loan_term | Loan term (years) |
| cibil_score | Credit score (300–900) |
| residential_assets_value | Value of residential assets owned |
| commercial_assets_value | Value of commercial assets owned |
| luxury_assets_value | Value of luxury assets owned |
| bank_asset_value | Value of bank assets/deposits |
| **loan_status** | **Target**: Approved / Rejected |

Class balance: **2,656 approved (62%) vs. 1,613 rejected (38%)** — mild
imbalance, not severe.

No missing values and no duplicate rows were found, but column names and
every text value had **leading whitespace** (e.g. `" Graduate"`,
`" no_of_dependents"`) — a common artifact of CSV exports that silently
breaks exact-match filtering/grouping if not cleaned.

## 3. Data Cleaning & Preprocessing

1. **Stripped whitespace** from all column names and string values (fixes
   the leading-space issue above).
2. **Dropped `loan_id`** — not predictive.
3. **No missing-value imputation needed** for this run, but the pipeline
   still includes a median/mode fallback so it's robust to new data with
   nulls.
4. **Feature engineering**:
   - `total_assets_value` = sum of residential + commercial + luxury + bank
     asset values (a single wealth signal instead of 4 correlated columns).
   - `loan_to_income_ratio` = loan_amount / income_annum (affordability).
   - `loan_to_assets_ratio` = loan_amount / total_assets_value (collateral
     coverage).
   - Log-transformed (`log1p`) the large, right-skewed monetary columns
     (income, loan amount, total assets) to reduce outlier influence.
5. **Encoding**: `education` and `self_employed` mapped to 0/1;
   `loan_status` mapped to 1 (Approved) / 0 (Rejected).
6. **Scaling**: numeric features standardized for Logistic Regression.
7. **Split**: 80/20 stratified split → 3,415 train / 854 test rows.

## 4. Models Trained

- **Logistic Regression** (linear baseline, interpretable)
- **Random Forest** (300 trees, max depth 8)

Evaluated with 5-fold stratified cross-validation plus a held-out test set.

## 5. Evaluation Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.910 | **1.000** |
| Precision | 0.916 | 1.000 |
| Recall | 0.942 | 1.000 |
| F1-score | 0.929 | 1.000 |
| ROC-AUC | 0.974 | 1.000 |
| 5-fold CV accuracy | 0.910 ± 0.020 | 0.998 ± 0.001 |

**Confusion matrix — Random Forest**, on the 854-row test set:

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 323 | 0 |
| **Actual Approved** | 0 | 531 |

Zero misclassifications.

## 6. What Drives the Predictions

Both models agree completely, and the signal is very concentrated:

- **`cibil_score` (credit score) accounts for ~83% of Random Forest's total
  feature importance** and has by far the largest Logistic Regression
  coefficient (+4.20). Everything else is a minor adjustment on top of it.
- `loan_term`, `loan_to_income_ratio`, and `loan_to_assets_ratio` contribute
  modestly.
- `education`, `self_employed`, and `no_of_dependents` contribute almost
  nothing to either model.

## 7. Explanation of Model Performance — including an important caveat

**Random Forest achieved a perfect test score (100% accuracy, F1 = 1.0),**
and Logistic Regression was close behind at 91% accuracy / F1 = 0.93. That
Random Forest result deserves an honest flag rather than a victory lap:

- **This dataset's approval decision is almost entirely rule-based on
  `cibil_score` alone.** A simple check confirms it directly: classifying
  every applicant with `cibil_score >= 550` as "Approved" and everyone else
  as "Rejected" — with *no other feature at all* — already gets **95.4%
  accuracy** on the full dataset. Approved applicants average a CIBIL score
  of ~703; rejected applicants average ~429, with limited overlap.
- That's a strong sign this is a **synthetically generated dataset** (common
  for Kaggle practice datasets) where the label was created from a
  threshold-style rule rather than messy real-world underwriting decisions.
  A Random Forest, which can carve out an arbitrarily sharp decision
  boundary, easily learns that threshold and gets every test example right.
- **Logistic Regression's 91%/0.93-F1 is the more realistic number to trust**
  here — it's a smooth linear model, so it can't perfectly reproduce a hard
  cutoff, and its errors (14% miss rate on rejections, per the classification
  report) show where a single linear boundary underfits a threshold rule.
- **Practical implication**: if this dataset is meant to stand in for a
  real-world lending problem, don't expect a 100%-accurate model to survive
  contact with real applicants — real credit decisions involve noise,
  edge cases, and factors not captured by a single score. Treat the
  Random Forest's perfect score as a property of *this particular
  synthetic dataset*, not evidence the problem is "solved." If you have
  access to a less clean, real-world version of this data, I'd expect
  both models' accuracy to drop into a more typical 80–90% range.

## 8. Files

- `loan_approval_v2.py` — full pipeline (cleaning, training, evaluation)
- `model_evaluation.png` — confusion matrix, ROC curve, feature importance,
  metric comparison
- `summary.json` — machine-readable summary of results
