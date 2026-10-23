# Push Version 4.4 dataset ZIP directly from Google Colab

Repository: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents
Branch: `realtime-v4.4`
Target folder: `data/realtime_v4_4/`

## 1. Locate the ZIP in Colab

Update `ZIP_PATH` in the cell below to the exact location of your dataset ZIP. Example: `/content/dataset_splits_audit1.zip`. This uploads the ZIP as-is; it does not extract it.

## 2. Add a GitHub token to Colab Secrets

In Colab, open the **key icon (Secrets)** and create a secret named `GITHUB_TOKEN`. Give the token repository Contents read/write permission and enable notebook access to the secret. Never print or commit the token.

## 3. Run this cell

```python
from google.colab import userdata
from pathlib import Path
import subprocess
import os
import shutil

REPO_URL = "https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents.git"
BRANCH = "realtime-v4.4"
ZIP_PATH = Path("/content/dataset_splits_audit1.zip")  # CHANGE this to your ZIP's actual path
CLONE_DIR = Path("/content/rt-language-switching-repo")
DEST_REL = Path("data/realtime_v4_4")

if not ZIP_PATH.is_file():
    raise FileNotFoundError(
        f"ZIP not found: {ZIP_PATH}. Check the path and list files with: !find /content -name '*.zip'"
    )

size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"ZIP: {ZIP_PATH.name} ({size_mb:.2f} MB)")
if size_mb >= 100:
    raise ValueError(
        "GitHub blocks individual files of 100 MB or more. Use Git LFS or an approved external dataset host."
    )

token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError("Add GITHUB_TOKEN in Colab Secrets and enable notebook access.")

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
target_zip = dest / ZIP_PATH.name
shutil.copy2(ZIP_PATH, target_zip)

subprocess.run(["git", "-C", str(CLONE_DIR), "config", "user.name", "Colab Dataset Upload"], check=True)
subprocess.run(["git", "-C", str(CLONE_DIR), "config", "user.email", "colab-dataset-upload@users.noreply.github.com"], check=True)
subprocess.run(["git", "-C", str(CLONE_DIR), "add", str(DEST_REL)], check=True)

status = subprocess.run(
    ["git", "-C", str(CLONE_DIR), "status", "--short"],
    check=True, capture_output=True, text=True
).stdout

if not status.strip():
    print("No changes to commit; this ZIP may already be uploaded.")
else:
    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "commit", "-m", f"Add Version 4.4 dataset ZIP: {ZIP_PATH.name}"],
        check=True, capture_output=True, text=True
    )
    result = subprocess.run(
        ["git", "-C", str(CLONE_DIR), "push", "origin", BRANCH],
        check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print("ZIP pushed successfully to", BRANCH)

print("Branch: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents/tree/realtime-v4.4")
print("Dataset folder: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents/tree/realtime-v4.4/data/realtime_v4_4")
```

## Notes
- If you don't know the ZIP path, run `!find /content -type f -name '*.zip'` in Colab.
- This commits the ZIP directly without extracting or modifying its contents.
- GitHub rejects individual files larger than or equal to 100 MB; use Git LFS or an approved external dataset host for larger archives.
- Do not upload secrets, private caller transcripts, or data whose license does not permit redistribution.
