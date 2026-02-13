"""
Model training module with MLflow experiment tracking.

This module implements the complete ML training pipeline including:
- Model initialization and training
- MLflow parameter and metric logging
- Artifact generation (confusion matrix, classification report, scaler)
- Model registration to MLflow Model Registry
"""

import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_processor import load_and_preprocess_data


def train_model(dataset_name='iris', C=1.0, penalty='l2', random_state=42):  # noqa: N803
    """
    Train a classification model with MLflow experiment tracking.

    This function performs the complete ML workflow:
    1. Loads and preprocesses data
    2. Trains a Logistic Regression model
    3. Logs parameters, metrics, and artifacts to MLflow
    4. Registers the model in MLflow Model Registry with tags and description

    All artifacts are logged to MLflow including:
    - Trained model (sklearn format)
    - Confusion matrix visualization
    - Classification report
    - Fitted scaler for production inference

    Args:
        dataset_name (str): Name of the sklearn dataset ('iris', 'wine', 'breast_cancer').
            Default is 'iris'.
        C (float): Inverse regularization strength for Logistic Regression.
            Smaller values specify stronger regularization. Default is 1.0.
        penalty (str): Regularization type ('l1' or 'l2'). Default is 'l2'.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        None. All outputs are logged to MLflow tracking server.

    Example:
        >>> train_model(dataset_name='iris', C=1.0, penalty='l2')
        # Trains model and logs to MLflow experiments

    Notes:
        - Requires MLflow tracking server to be running
        - Creates/uses experiment named "{dataset_name}_classification"
        - Automatically registers model with version management
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(dataset_name, random_state=random_state)  # noqa: N806

    # Set up MLflow
    mlflow.set_experiment(f"{dataset_name}_classification")
    with mlflow.start_run(run_name=f"{dataset_name}_logreg"):
        # Initialize and train model
        # Use solver that supports the chosen penalty type
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(
            C=C, 
            penalty=penalty, 
            solver=solver,
            random_state=random_state, 
            max_iter=1000
        )
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("solver", solver)
        mlflow.log_param("random_state", random_state)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Classification report
        report = classification_report(y_test, y_pred)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        # Save and log scaler
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing_artifacts")
        os.remove(scaler_path)

        # Log and register model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ClassificationModel"
        )

        # Add model description and tags
        client = mlflow.tracking.MlflowClient()
        model_version = model_info.registered_model_version
        client.update_model_version(
            name="ClassificationModel",
            version=model_version,
            description=f"Logistic Regression classifier trained on {dataset_name} dataset with C={C}, penalty={penalty}. F1-score: {f1:.4f}"
        )
        client.set_model_version_tag(
            name="ClassificationModel",
            version=model_version,
            key="dataset",
            value=dataset_name
        )
        client.set_model_version_tag(
            name="ClassificationModel",
            version=model_version,
            key="algorithm",
            value="LogisticRegression"
        )
        client.set_model_version_tag(
            name="ClassificationModel",
            version=model_version,
            key="f1_score",
            value=f"{f1:.4f}"
        )

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    train_model(dataset_name='iris')
