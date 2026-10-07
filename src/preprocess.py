import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    df = df.copy()
    df['Amount_log'] = np.log(df['Amount'] + 1)
    df['Hour'] = (df['Time'] % (24 * 3600)) // 3600
    df['Time_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Time_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    return df

def split_and_scale(df, test_size=0.2, random_state=42):
    feature_cols = [c for c in df.columns if c not in ['Class', 'Time']]
    X, y = df[feature_cols], df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=feature_cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler