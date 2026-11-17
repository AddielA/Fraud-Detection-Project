# Backlog

Tickets are grouped by priority. Work top to bottom: later tickets build on earlier ones.
Completed tickets are marked ✅ and keep a short note of what was done.

| ID | Ticket | Priority | Effort | Depends on |
|---|---|---|---|---|
| FD-1 | ✅ ~~Pin package versions~~ | Required | 15 min | |
| FD-2 | ✅ ~~Make scripts runnable from any folder~~ | Required | 30 min | |
| FD-3 | ✅ ~~Write the README~~ | Required | 1–2 hrs | FD-1, FD-2 |
| FD-4 | Move repeated results printout into `evaluate.py` | Optional | 30 min | |
| FD-5 | Tune decision thresholds on the validation set | Optional | 1–2 hrs | |
| FD-6 | Business-cost analysis | Optional | 2–3 hrs | FD-5 |
| FD-7 | Cross-validation for the top models | Optional | 1–2 hrs | |
| FD-8 | Results charts (PR curves, confusion matrices) | Stand-out | 1–2 hrs | |
| FD-9 | Explain predictions with SHAP | Stand-out | 1–2 hrs | |
| FD-10 | Time-based train/test split | Stand-out | 1–2 hrs | |
| FD-11 | Confidence intervals on model scores | Stand-out | 1 hr | |
| FD-12 | Unit tests + GitHub Actions CI | Stand-out | 2–3 hrs | FD-1, FD-2 |
| FD-13 | `predict.py` for scoring new transactions | Stand-out | 1 hr | |
| FD-14 | Interactive demo app | Stand-out | 3–5 hrs | FD-6, FD-13 |
| FD-15 | Dockerfile | Stand-out | 1 hr | FD-1 |

---

## Required

These make the repo usable by someone other than you. Finish them before sharing the GitHub link.

### FD-1: Pin package versions ✅ Complete (2026-11-17)

Every package in `requirements.txt` is pinned to the version installed in the project venv, and `joblib` was added. The pins require Python 3.9 (numpy 2.0.2 doesn't support newer versions); note this in the README.

### FD-2: Make scripts runnable from any folder ✅ Complete (2026-11-17)

`data_loader.py` defines `Root` (the project folder, found from the file's own location), and every data and model path in `src/` is built from it. The scripts work from the project root or from inside `src/`, and renaming the project folder no longer breaks them.

### FD-3: Write the README ✅ Complete (2026-11-17)

`README.md` covers the overview, results table and findings, dataset, setup, usage, approach, limitations and project structure. The setup steps were tested in a fresh clone; they include creating `data/raw/` and `models/`, which git does not track.

---

## Optional

Improvements to the analysis itself.

### FD-4: Move repeated results printout into `evaluate.py`

The "Frauds caught / missed / False alarms" block and classification report are copied into eight training scripts.

Steps:
1. Add `print_report(model, X_test, y_test)` to `evaluate.py` that prints the metrics, confusion-matrix counts and classification report.
2. Replace the copied block in each training script with one call.
3. Rerun each script and confirm the output is unchanged.

Done when: the printout exists in one place.

### FD-5: Tune decision thresholds on the validation set

Every model flags fraud when its score passes 0.5. A different cutoff can trade false alarms for caught fraud, and some of the gap between models may just be threshold placement.

Steps:
1. Use `split_train_val()` to get a validation set.
2. For each supervised model, compute `predict_proba` on the validation set and sweep thresholds from 0.01 to 0.99.
3. Pick the threshold with the best F1 (or the lowest cost, once FD-6 exists).
4. Save the chosen threshold next to each model and use it in `compare_models.py`.

Done when: the comparison reports results at both 0.5 and the tuned threshold. Pick thresholds on the validation set only, never the test set.

### FD-6: Business-cost analysis

The original goal of `cost_sensitive.py`: measure models in dollars, not just scores.

Steps:
1. Define costs: a missed fraud costs the transaction's `Amount`; a false alarm costs a fixed review fee (for example $5–$10); a caught fraud costs the review fee.
2. Write `cost_analysis.py` that computes total cost for each model on the test set.
3. Add a baseline: the cost of flagging nothing (all fraud gets through).
4. Using the validation set, pick each model's threshold to minimize cost (this extends FD-5).
5. Report savings versus the baseline, for example "XGBoost saves $X compared with no model".

Done when: the comparison ends with a dollar figure per model. This supports putting "cost-sensitive" back on the resume.

### FD-7: Cross-validation for the top models

Results come from one train/test split with 98 test frauds, so the top three models are a few frauds apart.

Steps:
1. Use `StratifiedKFold(n_splits=5)` on the full dataset.
2. For XGBoost, LightGBM and Random Forest + SMOTE, record PR-AUC, precision and recall on each fold. Fit the scaler and SMOTE inside each fold, not before splitting.
3. Report the mean and standard deviation for each.

Done when: you can say whether XGBoost's lead holds across folds.

---

## Stand-out

These go beyond a standard fraud-detection project. FD-8, FD-9 and FD-10 are the strongest for the effort.

### FD-8: Results charts

The README will have only tables. Charts make the results readable at a glance and put Matplotlib/Seaborn (listed on the resume) back into the project.

Steps:
1. Write `plot_results.py` that saves PNGs to a new `figures/` folder.
2. Plot precision-recall curves for all supervised models on one chart.
3. Plot a confusion matrix for the top model.
4. Embed both in the README.

Done when: the README shows the PR curves.

### FD-9: Explain predictions with SHAP

Fraud teams need to know why a transaction was flagged, not just that it was. SHAP shows which features pushed each prediction toward fraud.

Steps:
1. Add `shap` to `requirements.txt`.
2. Write `explain_model.py` that loads the XGBoost model and runs `shap.TreeExplainer` on the test set.
3. Save a summary plot (which features matter overall).
4. Save a waterfall plot for one caught fraud and one false alarm, and describe what drove each in the README.

Done when: the README includes an explanation of at least one flagged transaction.

### FD-10: Time-based train/test split

The current split is random, so the model trains on transactions from after some test transactions. Real fraud systems train on the past and predict the future.

Steps:
1. Add `split_by_time()` to `preprocess.py`: sort by `Time`, use the first 80% for training and the last 20% for testing.
2. Retrain XGBoost and one other model with it.
3. Compare scores with the random split.

Done when: the README reports both and explains the difference. A drop in scores is still a good result: it shows the random split was optimistic.

### FD-11: Confidence intervals on model scores

A lighter alternative to FD-7 that answers "is the winner really better?"

Steps:
1. Resample the test set with replacement 1,000 times.
2. Compute each model's PR-AUC on every resample.
3. Report the 95% interval (2.5th to 97.5th percentile) per model.

Done when: the scoreboard shows ranges such as "0.88 (0.83–0.92)", and the README says whether the top models' ranges overlap.

### FD-12: Unit tests + GitHub Actions CI

Your resume lists unit testing and GitHub Actions; this puts both in a project someone can see.

Steps:
1. Add `pytest` to a `requirements-dev.txt`.
2. Write tests in `tests/` using a small synthetic DataFrame (not the real CSV):
   - `engineer_features` adds the four expected columns.
   - `split_and_scale` keeps the fraud ratio in both splits, and the scaler is fit on training data only.
   - `get_class_weights` returns the right ratio.
   - `evaluate_model` returns all five metrics between 0 and 1.
3. Add `.github/workflows/tests.yml` to run `pytest` on every push.
4. Add the CI status badge to the README.

Done when: pushing to GitHub runs the tests automatically and the badge is green.

### FD-13: `predict.py` for scoring new transactions

Right now models can only be scored inside the training and comparison scripts.

Steps:
1. Write `predict.py` that takes a CSV path, applies `engineer_features` and the saved scaler, loads the chosen model, and writes a CSV with a fraud probability and flag for each row.
2. Use the tuned threshold from FD-5 if it exists.

Done when: `python src/predict.py some_transactions.csv` produces scored output.

### FD-14: Interactive demo app

A live demo is the easiest way for a recruiter to see the project without running code.

Steps:
1. Add `streamlit` to `requirements.txt`.
2. Build `app.py` with:
   - a model picker,
   - a threshold slider showing caught fraud, false alarms and total cost (from FD-6) updating live,
   - a way to score sample transactions using `predict.py`.
3. Deploy it free on Streamlit Community Cloud and link it from the README and resume.

Done when: there is a public link someone can open in a browser.

### FD-15: Dockerfile

Your resume lists Docker. A Dockerfile lets anyone run the project with one command, with no venv or `libomp` setup.

Steps:
1. Write a `Dockerfile` from `python:3.9-slim` that installs `libgomp1` and `requirements.txt`, then copies `src/`.
2. Mount `data/` as a volume so the dataset stays outside the image.
3. Add the build and run commands to the README.

Done when: `docker run` reproduces the results table.
