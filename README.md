# Credit Card Fraud Detection - Version 1.0.0 #

## Overview ##

A machine learning pipeline that detects fraudulent credit card transactions in a
dataset where only 0.172% of 284,807 transactions are fraud. It compares 10 models
across 6 algorithms, from Logistic Regression to XGBoost, handling the extreme
class imbalance with class weighting and SMOTE.

The best model, XGBoost, caught 82 of 98 frauds in the held-out test set with only
9 false alarms among 56,962 transactions (PR-AUC 0.88). Models are ranked by PR-AUC
rather than accuracy, since a model that flags nothing would still be 99.8% accurate.

## Results ##

All models are scored on the same held-out test set: 56,962 transactions, 98 of them fraud.
Sorted by PR-AUC, the main ranking metric.

| Model | Frauds caught (of 98) | False alarms | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | **82** | **9** | **0.901** | **0.837** | **0.868** | **0.883** |
| Random Forest + SMOTE | 79 | 9 | 0.898 | 0.806 | 0.849 | 0.873 |
| LightGBM | 81 | 10 | 0.890 | 0.827 | 0.857 | 0.864 |
| Random Forest | 74 | 3 | 0.961 | 0.755 | 0.846 | 0.863 |
| Neural Network + SMOTE | 83 | 37 | 0.692 | 0.847 | 0.761 | 0.840 |
| Neural Network | 72 | 9 | 0.889 | 0.735 | 0.804 | 0.833 |
| Logistic Regression + SMOTE | 90 | 1,538 | 0.055 | 0.918 | 0.104 | 0.727 |
| Logistic Regression | 90 | 1,471 | 0.058 | 0.918 | 0.108 | 0.719 |
| Isolation Forest (all data) | 28 | 89 | 0.239 | 0.286 | 0.260 | 0.138 |
| Isolation Forest (normal only) | 24 | 112 | 0.176 | 0.245 | 0.205 | 0.101 |

What the results show:

- **Gradient boosting and Random Forest perform best.** XGBoost has the best F1 and PR-AUC,
  while plain Random Forest raises the fewest false alarms (3) at the cost of missing more fraud.
- **Logistic Regression catches the most fraud but is impractical.** It flags about 1,500
  normal transactions to catch 90 frauds, roughly 16 false alarms for every real one.
- **SMOTE helps only a little.** It raised PR-AUC by less than 0.01 for each model. For
  Random Forest and the Neural Network, it caught more fraud at the cost of more false alarms.
- **The unsupervised Isolation Forest performs poorly.** Without labels, it finds unusual
  transactions, and most unusual transactions here are not fraud.
- **ROC-AUC is misleading here.** Isolation Forest scores 0.95 ROC-AUC but only 0.14 PR-AUC.
  With so few frauds, ROC-AUC looks high for almost any model, which is why PR-AUC is used for ranking.

## Dataset ##

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
from Kaggle contains two days of transactions by European cardholders in September 2013.
For privacy, the original features were transformed with PCA into 28 anonymous columns
(`V1`–`V28`). Only `Time`, `Amount` and the label `Class` (1 = fraud) are in their original form.

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and
place it in `data/raw/`.

## Setup ##

Requires Python 3.9–3.12 (developed on 3.9).

1. Clone the repository and create the data and model folders:

   ```bash
   git clone https://github.com/AddielA/Fraud-Detection-Project.git
   cd Fraud-Detection-Project
   mkdir -p data/raw models
   ```

2. Create a virtual environment and install the pinned dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. On macOS, XGBoost and LightGBM also need the OpenMP runtime:

   ```bash
   brew install libomp
   ```

4. Download `creditcard.csv` (see [Dataset](#dataset)) into `data/raw/`.

## Usage ##

Train the models first. Each training script prints its results and saves the model to `models/`:

```bash
python src/train_xgboost.py
```

To train all ten models in one go:

```bash
for f in src/train_*.py; do python "$f"; done
```

Then compare them:

```bash
python src/compare_models.py
```

The comparison prints a scoreboard, a plain-language breakdown of each model, head-to-head
rankings, strengths and weaknesses, and a final verdict. It also saves the scores to
`data/processed/model_comparison.csv`.

Scripts can be run from any folder.

## Approach ##

**Preprocessing.** `Amount` is log-scaled, and `Time` (seconds since the first transaction)
is turned into hour of day, encoded as sine and cosine so 11 PM and midnight end up close together.
The data is split 80/20 into training and test sets, stratified so both keep the same fraud rate.
The scaler is fit on the training set only, so no test information leaks into training.

**Handling imbalance.** Three approaches are compared:
- **Class weighting:** mistakes on fraud cost the model more during training
  (`class_weight='balanced'`, or `scale_pos_weight` for XGBoost and LightGBM, about 578:1).
- **SMOTE:** creates synthetic fraud examples, applied to the training set only.
- **Unsupervised anomaly detection:** Isolation Forest learns what normal looks like without labels.

**Early stopping.** XGBoost and LightGBM hold out 20% of the training set as a validation set
and stop adding trees once validation PR-AUC stops improving. The test set is only used for the
final score.

**Evaluation.** Every model is scored on precision, recall, F1, ROC-AUC and PR-AUC, plus raw counts
of frauds caught and false alarms, which are easier to interpret than ratios.

## Limitations ##

- **One train/test split.** The test set has only 98 frauds, so the top three models are within a
  few frauds of each other. Cross-validation would show whether XGBoost's lead holds.
- **Default decision threshold.** Every model flags fraud at a 0.5 probability. Tuning the
  threshold could change the precision/recall balance, and some of the gap between models.
- **Limited feature engineering.** Most features are anonymized PCA components, so only `Time`
  and `Amount` can be engineered.

## Project Structure ##

```
src/
├── data_loader.py                       # Loads the CSV and defines the project root path
├── preprocess.py                        # Feature engineering, splitting, scaling and class weights
├── evaluate.py                          # Shared metrics for supervised and anomaly models
├── train_logistic_regression.py
├── train_logistic_regression_smote.py
├── train_random_forest.py
├── train_random_forest_smote.py
├── train_neural_network.py
├── train_neural_network_smote.py
├── train_xgboost.py
├── train_lightgbm.py
├── train_isolation_forest.py            # Trains both Isolation Forest variants
└── compare_models.py                    # Scores every saved model and explains the results
```
