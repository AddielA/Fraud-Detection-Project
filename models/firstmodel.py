import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

# Imbalanced-learn for SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# For saving models
import joblib

# Visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All libraries imported successfully!")
print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load Data

df = pd.read_csv('data/raw/creditcard.csv')

print(" Database Overview:")
print(f"Total Transactions: {len(df):,}")
print(f"Features: {df.shape[1]}")
print(f"Fraud Transactions: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
print(f"Normal Transactions: {(1-df['Class']).sum():,} ({(1-df['Class']).mean()*100:.3f}%)")

print("\n Feature Engineering...")
# Log-transformed amount for skewed distributions
df['Amount_log'] = np.log(df['Amount'] + 1)

# Hour of day
df['Hour'] = (df['Time'] % (24 * 3600)) // 3600

# Time-based features
df['Time_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Time_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

print("Added engineering features: Amount_log, Hour, Time_sin, Time_cos")

feature_cols = [col for col in df.columns if col not in ['Class', 'Time']]
x = df[feature_cols]
y = df['Class']

print(f"\n Feature matrix shape: {x.shape}")
print(f"Selected features ({len(feature_cols)}): {', '.join(feature_cols[:5])}...")

# Train-test split with stratification to maintain class distribution
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n- Train set: {len(x_train):,} samples")
print(f"- Fraud: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
print(f"- Test set: {len(x_test):,} samples")
print(f"- Fraud: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

# Verify it worked
print(f"\n Stratification check: Train fraud rate = {y_train.mean():.4f}, Test fraud rate = {y_test.mean():.4f}")