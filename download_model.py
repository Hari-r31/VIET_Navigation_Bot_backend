"""
download_model.py — Download Vosk speech recognition models for offline STT.

Usage:
    python download_model.py           # downloads English (small, ~40 MB)
    python download_model.py --all     # downloads en + hi (te model not public yet)
    python download_model.py --lang hi # downloads a specific language

Models are saved to:   backend/models/<folder>/
"""

import argparse
import os
import urllib.request
import zipfile

# --------------------------------------------------------------------------- #
#  Available Vosk models                                                       #
#  Browse full list at: https://alphacephei.com/vosk/models                   #
# --------------------------------------------------------------------------- #
MODELS = {
    "en": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "folder": "vosk-model-en",
        "desc": "English (small, ~40 MB)",
    },
    "hi": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
        "folder": "vosk-model-hi",
        "desc": "Hindi (small, ~42 MB)",
    },
    # Telugu model is not yet publicly released by Vosk.
    # If you have a custom te model, place it at:  models/vosk-model-te/
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_model(lang: str):
    if lang not in MODELS:
        print(f"✗ No model entry for '{lang}'. Available: {list(MODELS.keys())}")
        return

    info = MODELS[lang]
    dest_folder = os.path.join(MODELS_DIR, info["folder"])

    if os.path.isdir(dest_folder):
        print(f"✓ Model [{lang}] already exists at: {dest_folder}")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    zip_path = os.path.join(MODELS_DIR, f"model-{lang}.zip")

    print(f"↓ Downloading {info['desc']} …")
    print(f"  URL: {info['url']}")

    def _progress(block, block_size, total):
        downloaded = block * block_size
        if total > 0:
            pct = min(downloaded * 100 // total, 100)
            print(f"\r  {pct}% ({downloaded // 1024 // 1024} MB / {total // 1024 // 1024} MB)", end="", flush=True)

    urllib.request.urlretrieve(info["url"], zip_path, reporthook=_progress)
    print()

    print("  Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Vosk zips contain a top-level folder; rename it to our standard name
        top = zf.namelist()[0].split("/")[0]
        zf.extractall(MODELS_DIR)

    # Rename extracted folder to our standard name if needed
    extracted_path = os.path.join(MODELS_DIR, top)
    if extracted_path != dest_folder and os.path.isdir(extracted_path):
        os.rename(extracted_path, dest_folder)

    os.remove(zip_path)
    print(f"  ✓ Model [{lang}] saved to: {dest_folder}")


def main():
    parser = argparse.ArgumentParser(description="Download Vosk STT models.")
    parser.add_argument("--lang", default="en", help="Language code (en | hi)")
    parser.add_argument("--all", action="store_true", help="Download all available models")
    args = parser.parse_args()

    if args.all:
        for lang in MODELS:
            download_model(lang)
    else:
        download_model(args.lang)


if __name__ == "__main__":
    main()
