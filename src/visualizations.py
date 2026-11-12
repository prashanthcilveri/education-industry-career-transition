"""
Visualisations for the career-transition study.

Produces the same four plots as the academic report and saves them as PNGs:
model accuracy comparison, Random Forest confusion matrix, Random Forest ROC
curve, and Random Forest feature importance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_accuracy_comparison(accuracy_results: dict, save_path: str) -> None:
    plt.figure()
    plt.bar(accuracy_results.keys(), accuracy_results.values())
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, save_path: str,
                          title: str = "Confusion Matrix - Random Forest") -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(model, X_test, y_test, save_path: str) -> None:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(importance_df, save_path: str) -> None:
    plt.figure()
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
