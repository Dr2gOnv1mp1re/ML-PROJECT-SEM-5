"""
download_vg.py
──────────────
Downloads the Visual Genome dataset required for Layer 2 (Visual Knowledge).

What it downloads:
  - Image metadata
  - Objects, attributes, relationships JSON files

Note: VG images are shared with MS COCO so we skip re-downloading images.

Run:
    py data/scripts/download_vg.py
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR = Path("data/raw/visual_genome")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"

URLS = {
    "objects"           : BASE_URL + "objects.json.zip",
    "attributes"        : BASE_URL + "attributes.json.zip",
    "relationships"     : BASE_URL + "relationships.json.zip",
    "region_descriptions": BASE_URL + "region_descriptions.json.zip",
    "question_answers"  : BASE_URL + "question_answers.json.zip",
    "image_data"        : BASE_URL + "image_data.json.zip",
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
    print(f"  Done → {dest}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Visual Genome Downloader")
    print(f"Save location: {SAVE_DIR.resolve()}")
    print("=" * 60)

    for name, url in URLS.items():
        print(f"\n[{name}]")
        zip_file = download_file(url, SAVE_DIR)
        extract_zip(zip_file, SAVE_DIR)

    print("\n✅ Visual Genome download complete!")
    print(f"   Files → {SAVE_DIR.resolve()}")
