# Add this to any model training script

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("mlflow_runs/")
mlflow.set_experiment("CineIQ-Experiments")

# Track a full training run
with mlflow.start_run(run_name="hybrid_v1"):
    
    # Log hyperparameters
    mlflow.log_param("svd_factors", 100)
    mlflow.log_param("svd_epochs", 20)
    mlflow.log_param("w_collab", 0.5)
    mlflow.log_param("w_content", 0.3)
    mlflow.log_param("w_sentiment", 0.2)
    
    # Log metrics after evaluation
    mlflow.log_metric("svd_rmse", 0.87)
    mlflow.log_metric("svd_mae", 0.67)
    mlflow.log_metric("coverage", 0.73)   # % of catalog recommended
    
    # Save model artifact
    mlflow.sklearn.log_model(svd_model, "svd_model")
    
    print("Run logged to MLflow!")

# View experiments in browser:
# mlflow ui --backend-store-uri mlflow_runs/
# Open: http://localhost:5000