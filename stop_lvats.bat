@echo off
title Lvats - Stop
cd /d "%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_lvats.ps1" -ProjectRoot "%~dp0." -Port 8000
if errorlevel 1 pause
