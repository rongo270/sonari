#!/usr/bin/env python
"""
download_models.py - fetch everything the tools need (run once on a new machine).

setup.bat / setup.sh call this automatically. You can also run it yourself:
    python download_models.py              # turbo + base + demucs + test clip
    python download_models.py --skip-turbo # lighter setup (skip the 1.6 GB model)

Everything is cached inside ./models. Re-running is safe — already-downloaded
files are skipped.
"""
import os
import sys
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MODELS = PROJECT / "models"
INPUT = PROJECT / "input"
MODELS.mkdir(exist_ok=True)
INPUT.mkdir(exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def download_whisper(name: str) -> None:
    from faster_whisper import WhisperModel

    print(f"[whisper] downloading '{name}' (this can take a while)...")
    WhisperModel(name, device="cpu", compute_type="int8", download_root=str(MODELS))
    print(f"[whisper] '{name}' ready.")


def download_demucs(name: str = "htdemucs") -> None:
    try:
        from demucs.pretrained import get_model

        print(f"[demucs] downloading '{name}'...")
        get_model(name)
        print(f"[demucs] '{name}' ready.")
    except Exception as exc:
        print(f"[demucs] skipped ({exc}). Music/lyrics needs 'pip install demucs'.")


def download_sample() -> None:
    dst = INPUT / "jfk.flac"
    if dst.exists():
        return
    url = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
    try:
        print("[sample] downloading jfk.flac test clip...")
        urllib.request.urlretrieve(url, dst)
        print(f"[sample] saved {dst}")
    except Exception as exc:
        print(f"[sample] skipped ({exc}).")


def main() -> None:
    skip_turbo = "--skip-turbo" in sys.argv
    print("=" * 60)
    print(" Downloading models for Whisper Transcriber")
    print("=" * 60)
    if not skip_turbo:
        download_whisper("large-v3-turbo")  # recommended default model
    download_whisper("base")                # small + fast fallback
    download_demucs("htdemucs")             # music -> isolated vocals
    download_sample()
    print("\nAll set. Test it with:")
    print("    python transcribe.py input/jfk.flac --model base")


if __name__ == "__main__":
    main()
