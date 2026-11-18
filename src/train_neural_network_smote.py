import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

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
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f" Training set after SMOTE: {len(X_train_res):,} samples "
          f"({y_train_res.sum():,} fraud, {(y_train_res == 0).sum():,} normal)")
    
    nn = MLPClassifier(
        hidden_layer_sizes=(100, 50), # Two Hidden Layers
        random_state=42,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn.fit(X_train_res, y_train_res)

    # 3. Evaluate
    print_report(nn, X_test, y_test)

    # 4. Save
    joblib.dump(nn, Root / 'models' / 'neural_network_smote.pkl')
    joblib.dump(scaler, Root / 'models' / 'scaler.pkl')


if __name__ == '__main__':
    main()