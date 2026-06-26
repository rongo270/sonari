@echo off
REM Easy launcher for speech -> text. From the project root:  tools\transcribe input\jfk.flac
"%~dp0..\venv\Scripts\python.exe" "%~dp0transcribe.py" %*
