"""
download_vqa.py
───────────────
Downloads the VQA v2 dataset required for Layer 3 (NLP / Question Classification).

What it downloads:
  - Training questions + annotations
  - Validation questions + annotations

Run:
    py data/scripts/download_vqa.py
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR = Path("data/raw/vqa_v2")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "train_questions"    : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "val_questions"      : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "train_annotations"  : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "val_annotations"    : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path):
    filename = url.split("/")[-1]
    filepath = dest / filename
    if filepath.exists():
        print(f"  [skip] {filename} already exists.")
        return filepath
    print(f"  Downloading {filename} ...")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
    return filepath


def extract_zip(zip_path: Path, dest: Path):
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"  Done -> {dest}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("VQA v2 Downloader")
    print(f"Save location: {SAVE_DIR.resolve()}")
    print("=" * 60)

    for name, url in URLS.items():
        print(f"\n[{name}]")
        zip_file = download_file(url, SAVE_DIR)
        extract_zip(zip_file, SAVE_DIR)

    print("\n[SUCCESS] VQA v2 download complete!")
    print(f"   Files -> {SAVE_DIR.resolve()}")
