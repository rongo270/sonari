@echo off
REM Easy launcher for music -> lyrics. Usage:  lyrics "input\song.mp3"
"%~dp0venv\Scripts\python.exe" "%~dp0lyrics.py" %*
