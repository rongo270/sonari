@echo off
REM Easy launcher for speech -> text. Usage:  transcribe input\jfk.flac
"%~dp0venv\Scripts\python.exe" "%~dp0transcribe.py" %*
