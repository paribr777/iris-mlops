import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse

# ------------------ Main Script ------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/iris.csv", help="Path to dataset")
    parser.add_argument("--model_name", type=str, default="Iris_Classifier", help="MLFlow registered model name")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")

    # Load the latest model from MLFlow registry
    model_uri = f"models:/{args.model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv(args.data)
    X = df.drop(columns=["species"], errors="ignore")
    y = df["species"]

    # Evaluate
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"Evaluated model from MLflow registry '{args.model_name}' & Accuracy: {acc:.4f}")
