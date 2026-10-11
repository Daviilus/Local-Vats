param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($ProjectRoot)
$python = if ($env:LVATS_PYTHON) { $env:LVATS_PYTHON } else { Join-Path $root '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $env:APPDATA 'LobsterAI\runtimes\python-win\python.exe'
}
if (-not (Test-Path -LiteralPath $python)) { throw '未找到可用于读取 Lvats 统一路径的 Python。' }
$previousAppRoot = $env:ASR_APP_ROOT
try {
    $env:ASR_APP_ROOT = $root
    $paths = (& $python -c "import os, sys; sys.path.insert(0, os.environ['ASR_APP_ROOT']); import json, lvats_paths; print(json.dumps(lvats_paths.PATHS.public_dict()))") | ConvertFrom-Json
} finally {
    $env:ASR_APP_ROOT = $previousAppRoot
}
$log = Join-Path ([string]$paths.LOG_DIR) 'server_8000.out.log'
if (-not (Test-Path -LiteralPath $log)) {
    Write-Output "[Lvats] 还没有日志文件：$log"
    exit 0
}
Get-Content -LiteralPath $log -Tail 60 -Wait
