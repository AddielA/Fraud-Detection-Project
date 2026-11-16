import joblib
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_data
from preprocess import engineer_features, split_and_scale, split_train_val, get_class_weights
from evaluate import evaluate_model


def main():
    # 1. Load and prepare data
    df = engineer_features(load_data())
    X_train, X_test, y_train, y_test, _ = split_and_scale(df)

    # Early stopping uses a validation set taken from the training data.
    # The test set is only used for the final score.
    X_fit, X_val, y_fit, y_val = split_train_val(X_train, y_train)
    _, ratio = get_class_weights(y_fit)

    print("\n Data splits:")
    print(f" Training:   {len(X_fit):,} samples ({y_fit.sum()} fraud)")
    print(f" Validation: {len(X_val):,} samples ({y_val.sum()} fraud)")
    print(f" Test:       {len(X_test):,} samples ({y_test.sum()} fraud)")

    # 2. Train
    print(f"\n Training LightGBM (scale_pos_weight = {ratio:.0f})...")
    lgbm = LGBMClassifier(
        n_estimators=2000,          # upper limit, early stopping picks the real number
        learning_rate=0.05,
        num_leaves=31,
        # These two settings keep training stable when scale_pos_weight is large.
        reg_lambda=10,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,     # missed fraud is penalized more, like class_weight='balanced'
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric='average_precision',    # PR-AUC on the validation set
        # stop when validation PR-AUC hasn't improved in 50 trees
        callbacks=[early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)]
    )
    val_pr_auc = lgbm.best_score_['valid_0']['average_precision']
    print(f" Stopped at {lgbm.best_iteration_} trees "
          f"(validation PR-AUC = {val_pr_auc:.4f})")

    # Feature importance
    importances = pd.Series(lgbm.feature_importances_, index=X_fit.columns)
    print("\n Top 10 Features (number of splits):")
    print("=" * 50)
    print(importances.sort_values(ascending=False).head(10))
    print("=" * 50)

    # 3. Evaluate
    metrics = evaluate_model(lgbm, X_test, y_test)
    print("\n Results:")
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")

    y_pred = lgbm.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\n Frauds caught: {tp}")
    print(f" Frauds missed: {fn}")
    print(f" False alarms: {fp}")
    print(f" Normal transactions correctly passed: {tn}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    # 4. Save
    joblib.dump(lgbm, 'models/lightgbm.pkl')


if __name__ == '__main__':
    main()
