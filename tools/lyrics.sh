#!/usr/bin/env bash
# Easy launcher for music -> lyrics (macOS/Linux).  Usage:  ./tools/lyrics.sh "input/song.mp3"
cd "$(dirname "$0")/.."
exec venv/bin/python lyrics.py "$@"
