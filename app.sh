#!/usr/bin/env bash
# Launch the Sonari visual app (web UI). It opens in your browser.
cd "$(dirname "$0")"
exec venv/bin/python app.py "$@"
