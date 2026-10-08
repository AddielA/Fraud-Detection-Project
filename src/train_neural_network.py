import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_data
from preprocess import engineer_features, split_and_scale
from evaluate import evaluate_model
import warnings
warnings.filterwarnings('ignore')


def main():
    # 1. Load and prepare data
    df = engineer_features(load_data())
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    # 2. Train
    print(" Training Neural Network...")
    nn = MLPClassifier(
        hidden_layer_sizes=(100, 50), # Two Hidden Layers
        random_state=42,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn.fit(X_train, y_train)

    # 3. Evaluate
    metrics = evaluate_model(nn, X_test, y_test)
    print("\n Results:")
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")

    y_pred = nn.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\n Frauds caught: {tp}")
    print(f" Frauds missed: {fn}")
    print(f" False alarms: {fp}")
    print(f" Normal transactions correctly passed: {tn}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    # 4. Save
    joblib.dump(nn, 'models/neural_network.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')


if __name__ == '__main__':
    main()