@echo off
REM ============================================================
REM  Sonari - ONE-CLICK SETUP for WINDOWS
REM  Double-click this file on a new PC. It installs everything
REM  and downloads all the AI models.
REM ============================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Sonari - Windows setup
echo ============================================================
echo.

REM --- 1) Find a suitable Python (3.10 - 3.12) ---
set "PYEXE="
for %%V in (3.11 3.12 3.10) do (
  if not defined PYEXE (
    py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V"
  )
)
if not defined PYEXE (
  python --version >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
  echo ERROR: Python 3.10-3.12 was not found.
  echo Install Python 3.11 from https://www.python.org/downloads/
  echo ^(tick "Add Python to PATH" during install^), then run this file again.
  pause & exit /b 1
)
echo Using Python: %PYEXE%
echo.

REM --- 2) Create the virtual environment ---
if not exist venv (
  echo Creating virtual environment...
  %PYEXE% -m venv venv || (echo Failed to create venv. & pause & exit /b 1)
)
set "VPY=venv\Scripts\python.exe"

REM --- 3) Upgrade pip + antivirus-SSL helper (AVG/Avast/etc.) ---
echo Upgrading pip...
"%VPY%" -m pip install --upgrade pip
echo Installing antivirus-SSL helper (harmless if you have no antivirus)...
"%VPY%" -m pip install pip-system-certs

REM --- 4) Install dependencies ---
echo.
echo Installing dependencies (faster-whisper, demucs, torch)...
echo This is the long step - it downloads a few hundred MB. Please wait.
"%VPY%" -m pip install -r requirements.txt || (echo Dependency install failed. & pause & exit /b 1)

REM --- 5) Download the AI models ---
echo.
echo Downloading AI models (about 1.8 GB - one time)...
"%VPY%" download_models.py

echo.
echo ============================================================
echo   DONE!  Now try:
echo       transcribe input\jfk.flac --model base
echo       lyrics "input\your-song.mp3"
echo ============================================================
pause
