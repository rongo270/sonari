@echo off
REM ============================================================
REM  Double-click to create the "Sonari" app icon (shortcut)
REM  on your Desktop and in this folder. Safe to run again.
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcut.ps1"
echo.
pause
