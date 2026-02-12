# Implement a function load_and_preprocess_data(dataset_name) that loads a scikit-learn toy dataset (e.g., load_iris, load_wine, load_breast_cancer).
# Perform any necessary preprocessing (e.g., StandardScaler if applicable).
# Split the data into training and testing sets with train_test_split(..., random_state=42).
# Return the processed data, target, and the fitted scaler object.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

def load_and_preprocess_data(dataset_name='iris', test_size=0.2, random_state=42):
    # Select dataset loader
    loaders = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    dataset = loaders[dataset_name]()
    X, y = dataset.data, dataset.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler)