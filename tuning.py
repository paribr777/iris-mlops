import os
import json
import argparse
import pandas as pd
import joblib
import subprocess
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------ Arguments ------------------ #
parser = argparse.ArgumentParser(description="Train Iris classifier with MLFlow and DVC")
parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV file")
parser.add_argument("--metrics", type=str, default="metrics.json", help="Path to save metrics JSON")
parser.add_argument("--version", type=str, default=None, help="Dataset/model version")
args = parser.parse_args()

DATA_PATH = args.data
METRICS_PATH = args.metrics
VERSION = args.version

MODELS_DIR = "models"

# ------------------ Helper Functions ------------------ #
def impute_missing(df):
    """Impute missing values using mean of last 10 samples per species."""
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        for species in df["species"].unique():
            mask = df["species"] == species
            vals = df.loc[mask, col].copy()
            for i in range(len(vals)):
                if pd.isna(vals.iloc[i]):
                    vals.iloc[i] = vals.iloc[max(0, i-10):i].mean()
            df.loc[mask, col] = vals
    return df

def train_and_evaluate(data_path, params):
    """Train RandomForest and evaluate accuracy."""
    df = pd.read_csv(data_path)
    df = impute_missing(df)

    X = df.drop(columns=["species"], errors="ignore")
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, acc

# ------------------ Main Script ------------------ #
if __name__ == "__main__":
    # Set version
    if VERSION is None:
        try:
            VERSION = subprocess.check_output(
                ["git", "describe", "--tags"], stderr=subprocess.STDOUT
            ).decode("utf-8").strip()
        except Exception:
            VERSION = "v0"

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Iris_Classifier_Model")

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 4],
    }

    best_acc = 0
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(run_name=f"iris_{VERSION}") as run:
            mlflow.set_tag("dataset_version", VERSION)
            clf, acc = train_and_evaluate(DATA_PATH, params)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(clf, "model")

            print(f"Params: {params} & Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_params = params

    # Save best model and metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, f"iris_best_{VERSION}.joblib")
    joblib.dump(best_model, best_model_path)

    metrics = {"version": VERSION, "best_accuracy": best_acc, "best_params": best_params}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{VERSION}] Best Accuracy: {best_acc:.4f} & Params: {best_params}")

