@echo off
title Lvats 日志（Ctrl+C 或关闭窗口退出）
cd /d "%~dp0"
if not exist "state\server_8000.out.log" (
  echo [Lvats] 还没有日志文件（服务可能未以后台模式启动过）。
  pause
  exit /b 0
)
powershell -NoProfile -Command "Get-Content -Path 'state\server_8000.out.log' -Tail 60 -Wait"
