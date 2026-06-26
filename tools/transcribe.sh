#!/usr/bin/env bash
# Easy launcher for speech -> text (macOS/Linux).  Usage:  ./tools/transcribe.sh input/jfk.flac
cd "$(dirname "$0")/.."
exec venv/bin/python tools/transcribe.py "$@"
