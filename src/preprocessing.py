"""
Data loading and preprocessing for the career-transition study.

Reconstructed from Appendix A of the academic report
("Bridging the Gap Between Education and Industry", Cilveri Prashanth, 2025-2026),
with one deliberate, documented correction:

  * ORIGINAL (Appendix A): StandardScaler was fitted on the ENTIRE feature
    matrix before the train-test split -- a data-leakage bug.
  * HERE: the split happens first and the scaler is fitted on the training
    rows only, then applied to the test rows.

See PROJECT_REVIEW.md, section 3.9.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COLUMN = "Likely to Change Occupation"


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset CSV and drop rows with missing values.

    Mirrors Appendix A: ``data = pd.read_csv('dataset5k.csv')`` followed by
    ``data.dropna()``.
    """
    data = pd.read_csv(path)
    return data.dropna()


def encode_categoricals(data: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every object (text) column, as in Appendix A.

    A fresh encoder is used per column. (Appendix A reused a single encoder
    instance for all columns; the result is identical here because each column
    is transformed in place, but per-column encoders are safer.)
    """
    data = data.copy()
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = LabelEncoder().fit_transform(data[column])
    return data


def split_features_target(data: pd.DataFrame, target: str = TARGET_COLUMN):
    """Split a preprocessed frame into (X, y) on the documented target column."""
    X = data.drop(target, axis=1)
    y = data[target].astype(int)
    return X, y


def prepare_train_test(data: pd.DataFrame, target: str = TARGET_COLUMN,
                       test_size: float = 0.2, random_state: int = 42):
    """Full pipeline: encode -> split -> scale (scaler fitted on TRAIN only).

    Returns (X_train, X_test, y_train, y_test, feature_names).
    The split parameters (test_size=0.2, random_state=42) are those used in
    Appendix A of the report.
    """
    encoded = encode_categoricals(data)
    X, y = split_features_target(encoded, target)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # FIX (see module docstring): fit the scaler on the training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names
