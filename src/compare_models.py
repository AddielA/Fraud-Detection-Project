import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from data_loader import load_data
from preprocess import engineer_features, split_and_scale
from evaluate import evaluate_model
import warnings
warnings.filterwarnings('ignore')

Model_Files = {
    'Logistic Regression' : 'models/logistic_regression.pkl',
    'Logistic Regression + SMOTE': 'models/logistic_regression_smote.pkl',
    'Random Forest' : 'models/random_forest.pkl',
    'Random Forest + SMOTE': 'models/random_forest_smote.pkl',
    'Neural Network' : 'models/neural_network.pkl',
    'Neural Network + SMOTE': 'models/neural_network_smote.pkl',
    'Isolation Fores ALL': 'models/isolation_forest.pkl',
    'Isolation Fores NORMAL': 'models/isolation_forest_normal.pkl'
}

Config_Notes = {
    'Logistic Regression' : "class_weight='balanced' (missed fraud is penalized more)",
    'Logistic Regression + SMOTE': "SMOTE on training data + class_weight='balanced'",
    'Random Forest' : "class_weight='balanced' (same idea as logistic regression, missed fraud is more penalized)",
    'Random Forest + SMOTE': "SMOTE on training data + class_weight='balanced'",
    'Neural Network' : "no class weighing (MLPClassifier has no class_weight option)",
    'Isolation Forest (all data)': "unsupervised, fit on all training data, contamination=0.002",
    'Isolation Forest (normal only)': "fit on normal transactions only, contamination=0.002",
}

# Metrics used for ranking. ROC-AUC is shown but not ranked, because with ~0.17% fraud
# it looks great for any model used; PR-AUC is a better measurement for ranking.
Rank_Metrics = ['precision', 'recall', 'f1', 'pr_auc']
Metric_Meaning = {
    'precision' : "avoiding false alarms",
    'recall'    : "catching real fraud",
    'f1'        : "balancing precision and recall",
    'pr_auc'    : "ranking fraud above normal transactions"
}

def header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def load_models():
    models = {}
    for name, path in Model_Files.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            print(f" Warning: {path} not found, skipping {name}.")
    return models

def collect_results(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        y_pred  = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results[name] = {**metrics, 'tn' : tn, 'fp': fp, 'fn': fn, 'tp': tp} 
    return results

def print_scoreboard(results):
    header("1. SCOREBOARD")
    df = pd.DataFrame(results).T[['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']].astype(float)
    print(df.round(4).to_string())
    print("\n How to read this table:")
    print(" - Precision: of everything flagged as fraud, how much really was fraud.")
    print(" - Recall:    of all real fraud, how much did the model catch.")
    print(" - F1:        one number balancing precision and recall.")
    print(" - ROC-AUC:   how well the model ranks fraud above normal.")
    print(" - PR-AUC:    same idea, more honest under extreme imbalance. PR-AUC is the main ranking score.")
    return df

def explain_model(name, r):
    header(f"2. MODEL BREAKDOWN: {name}")
    total_fraud = r['tp'] + r['fn']
    flagged = r['tp'] + r['fp']

    print(f" Settings: {Config_Notes.get(name, 'n/a')}")
    print(f"\n Frauds caught: {r['tp']}")
    print(f" Frauds missed: {r['fn']}")
    print(f" False alarms: {r['fp']}")
    print(f" Normal transactions correctly passed: {r['tn']}")

    print(f"\n This shows us that {name} caught {r['tp']} of {total_fraud} real frauds "
        f"(recall = {r['recall']*100:.1f}%).")
    if flagged > 0:
        print(f" Based on the precision score, only {r['precision']*100:.1f}% of the "
            f"{flagged} transactions it flagged were actually fraud.")
    if r['tp'] > 0:
        print(f" In practical terms, it raised about {r['fp'] / r['tp']:.1f} false alarms "
            f"for every real fraud it caught.")

    # Rule-of-thumb interpretation (my thresholds, adjust them if you disagree)
    if r['recall'] >= 0.85 and r['precision'] < 0.30:
        print(" Overall: AGGRESSIVE. It catches most fraud but inflates data with false alarms.")
    elif r['precision'] >= 0.80 and r['recall'] < 0.70:
        print(" Overall: CAUTIOUS. Its alerts are trustworthy, but it lets a lot of fraud through.")
    elif r['precision'] >= 0.80 and r['recall'] >= 0.80:
        print(" Overall: BALANCED and strong on both sides.")
    else:
        print(" Overall: MIXED. It trades off in a way that isn't clearly aggressive or cautious.")

def rank_models(results):
    df = pd.DataFrame(results).T[Rank_Metrics].astype(float)
    ranks = df.rank(ascending=False, method='min').astype(int)
    ranks['avg_rank'] = ranks.mean(axis=1)
    ranks = ranks.sort_values(['avg_rank', 'pr_auc'])
    return df, ranks

def print_rankings(ranks):
    header("3. HEAD-TO-HEAD RANKINGS (1 = best)")
    print(ranks.to_string())
    print("\n Each metric is ranked separately, then averaged. A lower average rank is better.")
    print(" This weights all four metrics equally. A real fraud team might weigh recall higher.")

def print_strengths_weaknesses(df, ranks):
    header("4. STRENGTHS AND WEAKNESSES")
    n = len(ranks)
    for name in ranks.index:
        best = [m for m in Rank_Metrics if ranks.loc[name, m] == 1]
        worst = [m for m in Rank_Metrics if ranks.loc[name, m] == n]
        print(f"\n {name}")
        for m in best:
            print(f"  + Best at {m} ({df.loc[name, m]:.4f}): top performer at {Metric_Meaning[m]}.")
        for m in worst:
            print(f"  - Weakest at {m} ({df.loc[name, m]:.4f}): last place at {Metric_Meaning[m]}.")
        if not best and not worst:
            print("  ~ Never best or worst on any metric: a middle-of-the-pack performer.")

def print_what_worked(results, df):
    header("5. WHAT WORKED AND WHAT DIDN'T")

    best_p = df['precision'].idxmax()
    best_r = df['recall'].idxmax()
    if best_p != best_r:
        print(f" No single model won everything. {best_p} had the best precision "
            f"and {best_r} had the best recall.")
        print(" This is the precision/recall tradeoff: pushing one up usually pulls the other down.")
    else:
        print(f" {best_p} led on both precision and recall, which is unusual and a strong result.")

    if 'Neural Network' in results and len(results) > 1:
        others = [n for n in results if n != 'Neural Network']
        nn_recall = results['Neural Network']['recall']
        other_recall = sum(results[n]['recall'] for n in others) / len(others)
        print(f"\n The Neural Network's recall was {nn_recall:.3f}, versus an average of "
            f"{other_recall:.3f} for the class-weighted models.")
        if nn_recall < other_recall:
            print(" This is consistent with the NN having no class weighting, so it is not told "
                "that missing fraud is costly. (Consistent with, not proof of: other differences also matter.)")
        else:
            print(" The NN matched or beat the weighted models even without class weighting.")

    print("\n Worth noting: all models use the default 0.5 decision threshold. "
        "Part of what separates them may be threshold placement, not model quality.")


def print_verdict(df, ranks, n_fraud_test):
    header("6. FINAL VERDICT")
    winner = ranks.index[0]
    print(f" Best overall by average rank: {winner}")
    for m in Rank_Metrics:
        print(f"   {m}: {df.loc[winner, m]:.4f}")

    print("\n Caveats to keep in mind:")
    print(f" - The test set has only {n_fraud_test} frauds, so a difference of a few "
        f"frauds can move the scores noticeably.")
    print(" - Results come from one train/test split. Cross-validation would be more reliable.")
    print(" - Equal-weighted ranking is a choice. If missed fraud costs more than false alarms, "
        "recall deserves more weight.")


def main():
    df_raw = engineer_features(load_data())
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_raw)

    models = load_models()
    if len(models) < 2:
        print(" Need at least two saved models to compare. Train them first.")
        return

    results = collect_results(models, X_test, y_test)

    print_scoreboard(results)
    for name, r in results.items():
        explain_model(name, r)

    df, ranks = rank_models(results)
    print_rankings(ranks)
    print_strengths_weaknesses(df, ranks)
    print_what_worked(results, df)
    print_verdict(df, ranks, int(y_test.sum()))

    os.makedirs('data/processed', exist_ok=True)
    pd.DataFrame(results).T.to_csv('data/processed/model_comparison.csv')


if __name__ == '__main__':
    main()