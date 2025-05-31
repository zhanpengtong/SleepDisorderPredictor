import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import joblib

# Define base directory as the root of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIGURE_DIR = os.path.join(BASE_DIR, 'figures')

X_TEST_PATH = os.path.join(DATA_DIR, 'X_test.npy')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.npy')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')


# --- Function Definitions ---

def plot_roc_curve(fpr, tpr, title='ROC Curve'):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def evaluate_models(X_test, y_test, model_path):
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, predictions))

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plot_roc_curve(fpr, tpr)

    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm)


# --- Main Execution ---
if __name__ == "__main__":
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    evaluate_models(X_test, y_test, MODEL_PATH)
