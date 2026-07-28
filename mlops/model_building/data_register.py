import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.environ["HF_USERNAME"]
DATASET_REPO = f"{HF_USERNAME}/uci-credit-card-default"
DATA_PATH = "mlops/data/UCI_Credit_Card.csv"

api = HfApi(token=os.environ["HF_TOKEN"])
create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=DATA_PATH,
    path_in_repo="UCI_Credit_Card.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)
print(f"Dataset uploaded to {DATASET_REPO}")
