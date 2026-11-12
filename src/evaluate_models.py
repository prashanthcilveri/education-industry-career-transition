"""
Model evaluation helpers: accuracy, confusion matrix, classification report
and ROC AUC -- the metrics used in the academic report (Chapters 5 and 6).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate one fitted model and return its metrics as a dictionary."""
    y_pred = model.predict(X_test)
    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        # AUC is undefined if the test set happens to contain a single class.
        if len(np.unique(y_test)) == 2:
            result["roc_auc"] = roc_auc_score(y_test, y_prob)
    return result


def random_forest_feature_importance(model, feature_names) -> "pd.DataFrame":
    """Feature importances for a fitted Random Forest.

    NOTE: Appendix A of the report built the names with
    ``data.drop('Employment_Status', axis=1).columns`` -- a column that does not
    exist in this dataset (see PROJECT_REVIEW.md, section 3.8). This function
    takes the true feature names used at training time.
    """
    import pandas as pd

    return (
        pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )
