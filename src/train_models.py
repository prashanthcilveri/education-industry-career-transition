"""
Model training for the career-transition study.

The six classifiers and their settings are exactly those in Appendix A of the
academic report. Run end-to-end with:

    python -m src.train_models --data data/dataset5k.csv

The original dataset is not distributed with this repository (see data/README.md),
so this script requires a local copy of the CSV with the documented schema.
"""

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import load_dataset, prepare_train_test
from .evaluate_models import evaluate_model


def build_models() -> dict:
    """The six classifiers, configured as in Appendix A of the report."""
    return {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }


def train_all(X_train, y_train) -> dict:
    """Fit every model on the training data and return the fitted models."""
    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate all six models.")
    parser.add_argument("--data", default="data/dataset5k.csv",
                        help="Path to the dataset CSV (default: data/dataset5k.csv)")
    args = parser.parse_args()

    data = load_dataset(args.data)
    X_train, X_test, y_train, y_test, _ = prepare_train_test(data)
    models = train_all(X_train, y_train)

    for name, model in models.items():
        result = evaluate_model(name, model, X_test, y_test)
        print("\n====================================")
        print(f"Model: {result['model']}")
        print("Accuracy:", result["accuracy"])
        print("Confusion Matrix:\n", result["confusion_matrix"])
        print("Classification Report:\n", result["classification_report"])


if __name__ == "__main__":
    main()
