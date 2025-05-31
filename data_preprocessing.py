import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define base directory as the root of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIGURE_DIR = os.path.join(BASE_DIR, 'figures')

def preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Split 'Blood Pressure' column into 'Systolic' and 'Diastolic'
    blood_pressure = data['Blood Pressure'].str.split('/', expand=True)
    data['Systolic'] = pd.to_numeric(blood_pressure[0], errors='coerce')
    data['Diastolic'] = pd.to_numeric(blood_pressure[1], errors='coerce')
    data.drop(columns=['Blood Pressure'], inplace=True)

    # Define features and target
    X = data.drop('Sleep Disorder', axis=1)
    y = data['Sleep Disorder']

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Define preprocessing pipelines
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    # Transform and split the data
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)
    print("Preprocessing completed. Data shapes:", X_train.shape, X_test.shape)
