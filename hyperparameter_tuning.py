import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define base directory as the root of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIGURE_DIR = os.path.join(BASE_DIR, 'figures')

X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train.csv')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.csv')

def tune_hyperparameters(X_train, y_train):
    """
    Perform grid search to tune hyperparameters for a RandomForestClassifier.
    Returns the best model found.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    }
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def main(X_train, y_train):
    best_model = tune_hyperparameters(X_train, y_train)
    return best_model

if __name__ == "__main__":
    # Load preprocessed training data from CSV
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()  # Ensure y is a 1D array if needed
    model = main(X_train, y_train)
    print("Best model parameters:", model.get_params())
