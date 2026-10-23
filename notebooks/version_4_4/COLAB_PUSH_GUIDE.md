# Upload the Version 4.4 ZIP bundle from Google Colab

This guide uploads the ZIP file **as-is**. If the ZIP contains both model files and dataset files, nothing is extracted or changed.

Repository: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents
Branch: `realtime-v4.4`
Upload destination: `artifacts/version_4_4/archives/`

## 1. Find the ZIP in Colab

Run this to list ZIP files and sizes:

```python
from pathlib import Path
for p in Path("/content").rglob("*.zip"):
    print(f"{p} — {p.stat().st_size / (1024**2):.2f} MB")
```

## 2. Set up GitHub authentication

In Colab **Secrets** (key icon), add `GITHUB_TOKEN` with repository Contents read/write permission and enable notebook access. Do not print or commit the token.

## 3. Upload the ZIP directly

Set `ZIP_PATH` to the path printed in step 1. This cell commits the archive unchanged to the Version 4.4 branch.

```python
from google.colab import userdata
from pathlib import Path
import subprocess, os, shutil

REPO_URL = "https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents.git"
BRANCH = "realtime-v4.4"
ZIP_PATH = Path("/content/your_model_and_data_bundle.zip")  # CHANGE to the exact ZIP path
CLONE_DIR = Path("/content/rt-language-switching-repo")
DEST_REL = Path("artifacts/version_4_4/archives")

if not ZIP_PATH.is_file() or ZIP_PATH.suffix.lower() != ".zip":
    raise FileNotFoundError(f"Set ZIP_PATH to an existing ZIP file. Current value: {ZIP_PATH}")

size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"Archive: {ZIP_PATH.name} ({size_mb:.2f} MB)")
if size_mb >= 100:
    raise ValueError(
        "GitHub blocks individual files >=100 MB. Use Git LFS or an approved external dataset/model host."
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
shutil.copy2(ZIP_PATH, dest / ZIP_PATH.name)

subprocess.run(["git", "-C", str(CLONE_DIR), "config", "user.name", "Colab ZIP Upload"], check=True)
subprocess.run(["git", "-C", str(CLONE_DIR), "config", "user.email", "colab-zip-upload@users.noreply.github.com"], check=True)
subprocess.run(["git", "-C", str(CLONE_DIR), "add", str(DEST_REL)], check=True)

status = subprocess.run(
    ["git", "-C", str(CLONE_DIR), "status", "--short"],
    check=True, capture_output=True, text=True
).stdout

if not status.strip():
    print("No new changes detected; this ZIP may already be present.")
else:
    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "commit", "-m", f"Add Version 4.4 model and dataset bundle: {ZIP_PATH.name}"],
        check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "push", "origin", BRANCH],
        check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print("ZIP bundle pushed successfully!")

print("Branch: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents/tree/realtime-v4.4")
print("Archive folder: https://github.com/abhinavnautiyalDS/Real-Time-Language-Switching-for-Voice-Agents/tree/realtime-v4.4/artifacts/version_4_4/archives")
```

## Important
- Uploads the entire ZIP as a single file; it does not extract it.
- GitHub blocks individual files of 100 MB or larger. If the ZIP exceeds that limit, use Git LFS or an approved external storage/model hosting service.
- Confirm the archive contains no credentials, private call transcripts, or data/model assets whose licenses prohibit redistribution.
