import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.environ["HF_USERNAME"]
SPACE_REPO = f"{HF_USERNAME}/credit-card-default-app"
TOKEN = os.environ["HF_TOKEN"]

create_repo(
    SPACE_REPO, repo_type="space", space_sdk="streamlit",
    exist_ok=True, token=TOKEN
)
HfApi(token=TOKEN).upload_folder(
    folder_path="mlops/deployment",
    repo_id=SPACE_REPO,
    repo_type="space"
)
print(f"Application deployed to {SPACE_REPO}")
