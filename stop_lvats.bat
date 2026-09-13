@echo off
title Lvats - Stop
cd /d "%~dp0"
set PORT=8000
set PID=
if exist "state\lvats.pid" for /f "tokens=2 delims=:, " %%a in ('findstr /C:"\"pid\"" state\lvats.pid') do set "PID=%%a"
if exist "state\lvats.pid" for /f "tokens=2 delims=:, " %%a in ('findstr /C:"\"port\"" state\lvats.pid') do set "PORT=%%a"
echo [Lvats] 正在请求停止（端口 %PORT%，pid %PID%）...
curl.exe --ssl-no-revoke -s -m 4 -X POST https://127.0.0.1:%PORT%/api/shutdown
echo.
if "%PID%"=="" goto waitport
for /L %%i in (1,1,8) do (
  tasklist /FI "PID eq %PID%" 2>nul | find /I "%PID%" >nul
  if errorlevel 1 (
    echo [Lvats] 服务已停止（耗时约 %%i 秒）。
    exit /b 0
  )
  timeout /t 1 /nobreak >nul
)
echo [Lvats] 8 秒仍未退出，强制结束...
taskkill /F /PID %PID% >nul 2>&1
del "state\lvats.pid" >nul 2>&1
echo [Lvats] 停止完成。
exit /b 0
:waitport
for /L %%i in (1,1,8) do (
  curl.exe --ssl-no-revoke -s -m 2 https://127.0.0.1:%PORT%/api/health >nul 2>&1
  if errorlevel 1 (
    echo [Lvats] 服务已停止。
    exit /b 0
  )
  timeout /t 1 /nobreak >nul
)
echo [Lvats] 8 秒内未确认停止，请检查任务管理器。
