param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($ProjectRoot)

function Resolve-LvatsPython {
    if ($env:LVATS_PYTHON -and (Test-Path -LiteralPath $env:LVATS_PYTHON)) { return $env:LVATS_PYTHON }
    $venv = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) { return $venv }
    $runtime = Join-Path $env:APPDATA 'LobsterAI\runtimes\python-win\python.exe'
    if (Test-Path -LiteralPath $runtime) { return $runtime }
    throw '未找到可用于读取 Lvats 统一路径的 Python。'
}

$python = Resolve-LvatsPython
$previousAppRoot = $env:ASR_APP_ROOT
try {
    $env:ASR_APP_ROOT = $root
    $paths = (& $python -c "import os, sys; sys.path.insert(0, os.environ['ASR_APP_ROOT']); import json, lvats_paths; print(json.dumps(lvats_paths.PATHS.public_dict()))") | ConvertFrom-Json
} finally {
    $env:ASR_APP_ROOT = $previousAppRoot
}
$pidPath = Join-Path ([string]$paths.RUNTIME_DIR) 'lvats.pid'
$pidValue = $null
if (Test-Path -LiteralPath $pidPath) {
    try {
        $record = Get-Content -Raw -LiteralPath $pidPath | ConvertFrom-Json
        $pidValue = [int]$record.pid
        if ($record.port) { $Port = [int]$record.port }
    } catch {}
}

Write-Output "[Lvats] 正在请求停止服务（端口 $Port，PID $pidValue）…"
& curl.exe --ssl-no-revoke -s -m 4 -X POST "https://127.0.0.1:$Port/api/shutdown"
for ($i = 1; $i -le 8; $i++) {
    if ($pidValue -and (-not (Get-Process -Id $pidValue -ErrorAction SilentlyContinue))) {
        Write-Output "[Lvats] 服务已停止（用时约 $i 秒）。"
        exit 0
    }
    if (-not $pidValue) {
        & curl.exe --ssl-no-revoke -s -m 2 "https://127.0.0.1:$Port/api/health" *> $null
        if ($LASTEXITCODE -ne 0) { Write-Output '[Lvats] 服务已停止。'; exit 0 }
    }
    Start-Sleep -Seconds 1
}
if ($pidValue -and (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)) {
    Stop-Process -Id $pidValue -Force
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    Write-Output '[Lvats] 超时后已结束记录的 Lvats 进程。'
    exit 0
}
throw '8 秒内未确认服务停止；未结束任何未验证所有权的进程。'
