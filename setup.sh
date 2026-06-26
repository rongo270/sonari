#!/usr/bin/env bash
# ============================================================
#  Sonari - ONE-CLICK SETUP for macOS / Linux
#  Run on a new machine:   bash setup.sh
#  (or rename to setup.command to double-click on macOS)
#  Installs everything and downloads all the AI models.
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  Sonari - macOS / Linux setup"
echo "============================================================"
echo

# --- 1) Find a suitable Python (3.10 - 3.12) ---
PYEXE=""
for c in python3.11 python3.12 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PYEXE="$c"; break; fi
done
if [ -z "$PYEXE" ]; then
  echo "ERROR: Python 3.10-3.12 not found."
  echo "Install it, e.g. on macOS:   brew install python@3.11"
  exit 1
fi
echo "Using Python: $PYEXE ($($PYEXE --version 2>&1))"
echo

# --- 2) Create the virtual environment ---
if [ ! -d venv ]; then
  echo "Creating virtual environment..."
  "$PYEXE" -m venv venv
fi
VPY="venv/bin/python"

# --- 3) Upgrade pip ---
echo "Upgrading pip..."
"$VPY" -m pip install --upgrade pip

# --- 4) Install dependencies ---
echo
echo "Installing dependencies (faster-whisper, demucs, torch)..."
echo "This is the long step - it downloads a few hundred MB. Please wait."
"$VPY" -m pip install -r requirements.txt

# --- 5) Download the AI models ---
echo
echo "Downloading AI models (about 1.8 GB - one time)..."
"$VPY" tools/download_models.py

echo
echo "============================================================"
echo "  DONE!  Launch the app with  ./app.sh , or try:"
echo "      ./tools/transcribe.sh input/jfk.flac --model base"
echo "      ./tools/lyrics.sh \"input/your-song.mp3\""
echo "============================================================"
