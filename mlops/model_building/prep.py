import os
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

TARGET = "default.payment.next.month"
RANDOM_STATE = 42
HF_USERNAME = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/uci-credit-card-default"

api = HfApi(token=os.environ["HF_TOKEN"])
raw_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="UCI_Credit_Card.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
df = pd.read_csv(raw_path)

required = {"ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", TARGET}
if not required.issubset(df.columns):
    raise ValueError(f"Dataset schema is incomplete: {sorted(required - set(df.columns))}")

# Keep all columns in split files so demographics remain available for EDA/auditing.
train_val, test = train_test_split(
    df, test_size=0.20, random_state=RANDOM_STATE, stratify=df[TARGET]
)
train, validation = train_test_split(
    train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=train_val[TARGET]
)

output_dir = Path("mlops/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
for name, part in {"train": train, "validation": validation, "test": test}.items():
    path = output_dir / f"{name}.csv"
    part.to_csv(path, index=False)
    api.upload_file(
        path_or_fileobj=str(path), path_in_repo=f"processed/{name}.csv",
        repo_id=DATASET_REPO, repo_type="dataset"
    )
    print(name, part.shape, "default rate:", round(part[TARGET].mean(), 4))
