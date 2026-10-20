"""
download_coco.py
────────────────
Downloads the MS COCO 2017 dataset required for Layer 1 (Object Detection).

What it downloads:
  - val2017 images       (~1 GB)
  - annotations (JSON)   (~241 MB)
  (train2017 is optional / commented by default to save 18 GB disk & time)

Run:
    py data/scripts/download_coco.py
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR = Path("data/raw/coco")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "val2017_images"     : "http://images.cocodataset.org/zips/val2017.zip",
    "annotations"        : "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    # "train2017_images" : "http://images.cocodataset.org/zips/train2017.zip", # Uncomment if full 18GB training set is needed
}

# ── Helpers ───────────────────────────────────────────────────────────────────
class DownloadProgress(tqdm):
    """Progress bar for urllib.request.urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path):
    """Download a single file with a progress bar."""
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
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"  Done -> {dest}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MS COCO 2017 Downloader")
    print(f"Save location: {SAVE_DIR.resolve()}")
    print("=" * 60)

    for name, url in URLS.items():
        print(f"\n[{name}]")
        zip_file = download_file(url, SAVE_DIR)
        extract_zip(zip_file, SAVE_DIR)

    print("\n[SUCCESS] MS COCO download complete!")
    print(f"   Images -> {SAVE_DIR / 'val2017'}")
    print(f"   Labels -> {SAVE_DIR / 'annotations'}")
