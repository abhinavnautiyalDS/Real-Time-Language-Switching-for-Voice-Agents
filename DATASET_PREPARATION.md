# Dataset Preparation Colab

Branch: `dataset-prep/colab-notebook`

Notebook: `notebooks/01_dataset_split_from_zip.ipynb`

## What it does
1. Uploads/extracts `prepared_dataset.zip`.
2. Detects the combined data table.
3. Maps the current categories to 12 final labels: `as-IN`, `bn-IN`, `en-IN`, `gu-IN`, `hi-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `pa-IN`, `ta-IN`, `te-IN`, `none`.
4. Stops if the removed `marathi_hindi_mixed_script` category appears or if unknown categories/label conflicts are found.
5. Removes normalized duplicate texts and creates group-aware 80/10/10 train/validation/test splits.
6. Exports compressed splits, category counts, and a manifest.

## Run it
- Open the notebook from GitHub and choose **Open in Colab**, or clone the repository and upload the notebook to Colab.
- Run cells in order; upload `prepared_dataset.zip` when prompted.
- Review the detected file list and choose the correct combined table if there is more than one candidate.
- Download `processed_dataset_bundle.zip` and put its split files in the main repo's `data/processed/` folder.

## Important
This notebook prepares data only; it does not fine-tune BERT. It preserves provenance/review metadata where available and does not certify synthetic/unreviewed rows as verified. Do not push private call data or dataset text to a public repository unless redistribution is permitted.
