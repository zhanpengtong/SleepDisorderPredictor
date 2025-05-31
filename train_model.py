import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification


def train_models(X_train, y_train, X_test, y_test):
    # Check for NaN values and impute if any
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        print("NaN values detected, applying imputation.")
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model training completed. Accuracy: {accuracy:.2f}")

    return model, accuracy


if __name__ == "__main__":
    # Generate synthetic data for testing
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Introduce some NaN values artificially for testing
    X_train[::10] = np.nan

    # Train and test the model
    model, accuracy = train_models(X_train, y_train, X_test, y_test)
    print(f"Tested model accuracy: {accuracy:.2f}")
