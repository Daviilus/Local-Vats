@echo off
title Lvats Logs
cd /d "%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\view_lvats_log.ps1" -ProjectRoot "%~dp0."
if errorlevel 1 pause
