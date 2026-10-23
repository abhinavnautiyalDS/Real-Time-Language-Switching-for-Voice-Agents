# Push Version 4.4 dataset from Google Colab

Repository: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents
Branch: `realtime-v4.4`
Target folder: `data/realtime_v4_4/`

## 1. Prepare your files in Colab

Place the final files in a directory such as `/content/processed_dataset/`:
- `train.csv`
- `validation.csv`
- `test.csv`
- optional `dataset_manifest.json`

## 2. Add a GitHub token to Colab Secrets

In Colab, open the **key icon (Secrets)** and create a secret named `GITHUB_TOKEN`. Give the token repository Contents read/write access. Enable notebook access to that secret. Do not paste the token directly into a notebook cell or commit it.

## 3. Run this cell

```python
from google.colab import userdata
from pathlib import Path
import subprocess
import os
import shutil

REPO_URL = "https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents.git"
BRANCH = "realtime-v4.4"
SOURCE_DIR = Path("/content/processed_dataset")
CLONE_DIR = Path("/content/rt-language-switching-repo")
DEST_REL = Path("data/realtime_v4_4")

required = ["train.csv", "validation.csv", "test.csv"]
missing = [name for name in required if not (SOURCE_DIR / name).is_file()]
if missing:
    raise FileNotFoundError(f"Missing required dataset files in {SOURCE_DIR}: {missing}")

token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError("Add GITHUB_TOKEN in Colab Secrets and enable notebook access.")

# Use the token only in the process environment; never print it.
env = os.environ.copy()
env["GIT_TERMINAL_PROMPT"] = "0"

if CLONE_DIR.exists():
    shutil.rmtree(CLONE_DIR)

auth_url = REPO_URL.replace("https://", f"https://x-access-token:{token}@")
subprocess.run(
    ["git", "clone", "--branch", BRANCH, "--single-branch", auth_url, str(CLONE_DIR)],
    check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

dest = CLONE_DIR / DEST_REL
dest.mkdir(parents=True, exist_ok=True)

to_copy = required + ["dataset_manifest.json", "label_mapping.json"]
for name in to_copy:
    src = SOURCE_DIR / name
    if src.is_file():
        shutil.copy2(src, dest / name)

# Check sizes before committing; GitHub blocks individual files above 100 MB.
for path in sorted(dest.iterdir()):
    if path.is_file():
        mb = path.stat().st_size / (1024 * 1024)
        print(f"{path.name}: {mb:.2f} MB")
        if mb >= 100:
            raise ValueError(f"{path.name} is >=100 MB. Use Git LFS or an external dataset store.")

subprocess.run(["git", "-C", str(CLONE_DIR), "config", "user.name", "Colab Dataset Upload"], check=True)
subprocess.run(["git", "-C", str(CLONE_DIR), "config", "user.email", "colab-dataset-upload@users.noreply.github.com"], check=True)
subprocess.run(["git", "-C", str(CLONE_DIR), "add", str(DEST_REL)], check=True)

status = subprocess.run(
    ["git", "-C", str(CLONE_DIR), "status", "--short"],
    check=True, capture_output=True, text=True
).stdout
if not status.strip():
    print("No changes to commit; the repository already has these files.")
else:
    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "commit", "-m", "Add Version 4.4 dataset splits"],
        check=True, capture_output=True, text=True
    )
    # Use the authenticated remote for push; token is never printed.
    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "push", "origin", BRANCH],
        check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print("Dataset files pushed successfully to branch", BRANCH)

print("Branch URL: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents/tree/realtime-v4.4")
print("Dataset folder: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents/tree/realtime-v4.4/data/realtime_v4_4")
```

## Notes
- Change `SOURCE_DIR` if your processed files are stored somewhere else.
- This pushes only the named dataset files; it does not upload your entire Colab runtime.
- Never upload access tokens, private caller transcripts, or data whose license does not allow redistribution.
- GitHub blocks files larger than 100 MB. Use Git LFS or an approved external dataset host for large files.
- Check the GitHub repository's current licensing and dataset terms before publishing.
