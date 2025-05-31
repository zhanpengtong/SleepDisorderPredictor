import os
import numpy as np
import joblib
from data_preprocessing import preprocess_data
from feature_engineering import enhance_features
from train_model import train_models  # Ensure this function exists
from model_evaluation import evaluate_models  # Ensure this function exists

# Define base directory and relevant paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def main():
    try:
        # Step 1: Data preprocessing
        X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)
        print("Data preprocessing completed successfully.")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return

    try:
        # Step 2: Feature engineering
        X_train, X_test = enhance_features(X_train, X_test)
        print("Feature engineering completed successfully.")
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return

    try:
        # Step 3: Model training (expected to return a dictionary of model names and file paths)
        models = train_models(X_train, y_train, X_test, y_test)
        print("Model training completed successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    try:
        # Step 4: Model evaluation
        for model_name, model_path in models.items():
            print(f"Evaluating model: {model_name}")
            model = joblib.load(model_path)
            evaluate_models(X_test, y_test, model_path)  # Model path is passed, not model instance
        print("Model evaluation completed successfully.")
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

if __name__ == "__main__":
    main()
