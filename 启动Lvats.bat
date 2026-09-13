@echo off
title Lvats Launcher
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set ASR_PORT=8000
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_local_https.ps1"
if errorlevel 1 (
  echo [Lvats] HTTPS certificate setup failed.
  pause
  exit /b 1
)
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_lvats.ps1" -ProjectRoot "%~dp0." -Port 8000
if errorlevel 1 (
  pause
  exit /b 1
)
timeout /t 3 /nobreak >nul
exit /b 0