"""
Script to execute multiple MLflow experiments with different hyperparameters.
This demonstrates experiment tracking and comparison capabilities.
"""

import mlflow
from src.model_trainer import train_model

def run_multiple_experiments():
    """
    Execute multiple training runs with different hyperparameters to demonstrate
    MLflow experiment tracking and comparison.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Experiment configurations
    experiments = [
        {
            "dataset_name": "iris",
            "C": 1.0,
            "penalty": "l2",
            "description": "Baseline model with default L2 regularization"
        },
        {
            "dataset_name": "iris",
            "C": 0.1,
            "penalty": "l2",
            "description": "Stronger L2 regularization (lower C)"
        },
        {
            "dataset_name": "iris",
            "C": 10.0,
            "penalty": "l2",
            "description": "Weaker L2 regularization (higher C)"
        },
        {
            "dataset_name": "iris",
            "C": 1.0,
            "penalty": "l1",
            "description": "L1 regularization for feature selection"
        },
    ]
    
    print("Starting multiple experiment runs...")
    print(f"Total experiments to run: {len(experiments)}\n")
    
    for idx, config in enumerate(experiments, 1):
        print(f"{'='*60}")
        print(f"Experiment {idx}/{len(experiments)}: {config['description']}")
        print(f"Parameters: C={config['C']}, penalty={config['penalty']}")
        print(f"{'='*60}")
        
        try:
            train_model(
                dataset_name=config["dataset_name"],
                C=config["C"],
                penalty=config["penalty"],
                random_state=42
            )
            print(f"✓ Experiment {idx} completed successfully\n")
        except Exception as e:
            print(f"✗ Experiment {idx} failed: {str(e)}\n")
    
    print("="*60)
    print("All experiments completed!")
    print("View results at: http://localhost:5000")
    print("="*60)

if __name__ == "__main__":
    run_multiple_experiments()
