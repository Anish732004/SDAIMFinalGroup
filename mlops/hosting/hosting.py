import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.environ["HF_USERNAME"]
SPACE_REPO = f"{HF_USERNAME}/credit-card-default-app"
TOKEN = os.environ["HF_TOKEN"]

create_repo(
    SPACE_REPO, repo_type="space", space_sdk="docker",
    exist_ok=True, token=TOKEN
)
api = HfApi(token=TOKEN)
try:
    api.add_space_variable(repo_id=SPACE_REPO, key="HF_USERNAME", value=HF_USERNAME)
except Exception as e:
    print(f"Note: Could not set HF_USERNAME space variable: {e}")

# Inject ENV HF_USERNAME in Dockerfile before upload
dockerfile_path = "mlops/deployment/Dockerfile"
if os.path.exists(dockerfile_path):
    with open(dockerfile_path, "r", encoding="utf-8") as f:
        docker_lines = f.readlines()
    if not any("ENV HF_USERNAME" in line for line in docker_lines):
        docker_lines.insert(5, f"ENV HF_USERNAME={HF_USERNAME}\n")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.writelines(docker_lines)

# Ensure app.py uses actual HF_USERNAME
app_py_path = "mlops/deployment/app.py"
if os.path.exists(app_py_path):
    with open(app_py_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace('"YOUR_HF_USERNAME"', f'"{HF_USERNAME}"')
    with open(app_py_path, "w", encoding="utf-8") as f:
        f.write(content)

api.upload_folder(
    folder_path="mlops/deployment",
    repo_id=SPACE_REPO,
    repo_type="space"
)
print(f"Application deployed to {SPACE_REPO}")
