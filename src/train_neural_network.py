import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier

from data_loader import load_data, Root
from preprocess import engineer_features, split_and_scale
from evaluate import print_report
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
    print_report(nn, X_test, y_test)

    # 4. Save
    joblib.dump(nn, Root / 'models' / 'neural_network.pkl')
    joblib.dump(scaler, Root / 'models' / 'scaler.pkl')


if __name__ == '__main__':
    main()