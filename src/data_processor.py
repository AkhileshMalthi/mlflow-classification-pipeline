"""
Data loading and preprocessing module for classification models.

This module provides functionality to load scikit-learn toy datasets,
apply preprocessing transformations, and split data for training and testing.
"""

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(dataset_name="iris", test_size=0.2, random_state=42):
    """
    Load and preprocess a scikit-learn classification dataset.

    This function loads one of the built-in scikit-learn datasets, applies
    standardization scaling, and splits the data into training and testing sets.

    Args:
        dataset_name (str): Name of the dataset to load. Supported values are:
            'iris', 'wine', 'breast_cancer'. Default is 'iris'.
        test_size (float): Proportion of data to use for testing (0.0 to 1.0).
            Default is 0.2 (20% test, 80% train).
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing:
            - X_train_scaled (ndarray): Scaled training features
            - X_test_scaled (ndarray): Scaled testing features
            - y_train (ndarray): Training labels
            - y_test (ndarray): Testing labels
            - scaler (StandardScaler): Fitted scaler object for production inference

    Raises:
        ValueError: If dataset_name is not one of the supported datasets.

    Example:
        >>> X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('iris')
        >>> print(X_train.shape)
        (120, 4)
    """
    # Select dataset loader
    loaders = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    dataset = loaders[dataset_name]()
    X, y = dataset.data, dataset.target  # noqa: N806

    # Split data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features: fit on training data, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # noqa: N806
    X_test_scaled = scaler.transform(X_test)  # noqa: N806

    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
