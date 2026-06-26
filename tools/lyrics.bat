@echo off
REM Easy launcher for music -> lyrics. From the project root:  tools\lyrics "input\song.mp3"
"%~dp0..\venv\Scripts\python.exe" "%~dp0..\lyrics.py" %*
