import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from huggingface_hub import HfApi, create_repo, hf_hub_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

sys.path.append("mlops")
from feature_utils import TARGET, RAW_MODEL_FEATURES, engineer_features

RANDOM_STATE = 42
HF_USERNAME = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/uci-credit-card-default"
MODEL_REPO = f"{HF_USERNAME}/credit-card-default-model"
TOKEN = os.environ["HF_TOKEN"]


def load_split(name):
    path = hf_hub_download(
        repo_id=DATASET_REPO, filename=f"processed/{name}.csv",
        repo_type="dataset", token=TOKEN
    )
    frame = pd.read_csv(path)
    return engineer_features(frame[RAW_MODEL_FEATURES]), frame[TARGET]


def best_f1_threshold(y_true, probability):
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(f1[:-1].argmax())
    return float(thresholds[idx])


def metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")

models = {
    "Logistic Regression": Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, max_depth=12, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
    )
}

validation_results = []
test_results = []
trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_prob)
    val_m = metrics(y_val, val_prob, threshold)
    val_m["model"] = name
    validation_results.append(val_m)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_m = metrics(y_test, test_prob, threshold)
    test_m["model"] = name
    test_results.append(test_m)
    
    trained[name] = model

cols_order = ["model", "roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall", "threshold", "tn", "fp", "fn", "tp"]
validation_df = pd.DataFrame(validation_results)[cols_order].sort_values("roc_auc", ascending=False)
test_df = pd.DataFrame(test_results)[cols_order]

best_name = validation_df.iloc[0]["model"]
best_threshold = float(validation_df.iloc[0]["threshold"])
best_model = trained[best_name]

artifact_dir = Path("mlops/artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)
model_path = artifact_dir / "credit_default_model.joblib"
metadata_path = artifact_dir / "model_metadata.json"
comparison_path = artifact_dir / "validation_comparison.csv"
test_comparison_path = artifact_dir / "test_comparison.csv"

joblib.dump(best_model, model_path)
validation_df.to_csv(comparison_path, index=False)
test_df.to_csv(test_comparison_path, index=False)

metadata = {
    "model_name": best_name,
    "threshold": best_threshold,
    "raw_model_features": RAW_MODEL_FEATURES,
    "engineered_features": X_train.columns.tolist(),
    "validation_comparison": validation_df.to_dict(orient="records"),
    "test_comparison": test_df.to_dict(orient="records"),
    "test_metrics": test_df[test_df["model"] == best_name].to_dict(orient="records")[0]
}
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# Real experiment logging. A file store works in GitHub Actions and is uploaded as an artefact.
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit-card-default-risk")
with mlflow.start_run(run_name=best_name):
    mlflow.log_param("selected_model", best_name)
    mlflow.log_param("threshold", best_threshold)
    best_test = test_df[test_df["model"] == best_name].iloc[0]
    mlflow.log_metrics({f"test_{k}": float(v) for k, v in best_test.items() if k != "model"})
    mlflow.log_artifact(str(metadata_path))
    mlflow.log_artifact(str(comparison_path))
    mlflow.log_artifact(str(test_comparison_path))
    mlflow.sklearn.log_model(best_model, artifact_path="model")

api = HfApi(token=TOKEN)
create_repo(MODEL_REPO, repo_type="model", exist_ok=True, token=TOKEN)
for path in [model_path, metadata_path, comparison_path, test_comparison_path]:
    api.upload_file(
        path_or_fileobj=str(path), path_in_repo=path.name,
        repo_id=MODEL_REPO, repo_type="model"
    )
print("Selected model:", best_name)
print("Threshold:", round(best_threshold, 4))
print("Test metrics:", test_metrics)
