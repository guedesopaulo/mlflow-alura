import argparse
import mlflow
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import math

def parse_args():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Train an XGBoost model and log results with MLflow.")
    parser.add_argument("--learning_rate", type=float, default=0.2, help="Learning rate for XGBoost.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum depth of the trees.")
    parser.add_argument("--data_path", type=str, default="data/processed/casas.csv", help="Path to the input CSV file.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:///mnt/c/Users/Paulo/Documents/mlflow-alura/mlruns")

    # Load dataset
    df = pd.read_csv(args.data_path)
    X = df.drop("preco", axis=1)
    y = df["preco"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed)

    # Create DMatrix for XGBoost
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    # Define XGBoost parameters
    xgb_params = {
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "max_depth": args.max_depth
    }

    # Set MLflow experiment
    mlflow.set_experiment("house-prices-script")

    with mlflow.start_run():
        mlflow.xgboost.autolog()

        # Train the model
        xgb = xgboost.train(xgb_params, dtrain, evals=[(dtrain, "train")])

        # Predict and evaluate
        xgb_predicted = xgb.predict(dtest)
        mse = mean_squared_error(y_test, xgb_predicted)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, xgb_predicted)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")

if __name__ == "__main__":
    main()
