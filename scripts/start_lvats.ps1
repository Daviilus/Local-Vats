param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($ProjectRoot)
$state = Join-Path $root 'state'
$pidPath = Join-Path $state 'lvats.pid'
$stdoutPath = Join-Path $state 'server_8000.out.log'
$stderrPath = Join-Path $state 'server_8000.err.log'
$url = "https://127.0.0.1:$Port"
[IO.Directory]::CreateDirectory($state) | Out-Null

function Get-LvatsHealth {
    try {
        $json = & curl.exe --ssl-no-revoke -s -m 2 "$url/api/health" 2>$null
        if (($LASTEXITCODE -eq 0) -and $json) { return ($json | ConvertFrom-Json) }
    } catch {}
    return $null
}

function Resolve-LvatsExecutable {
    if ($env:LVATS_PYTHON) {
        $configured = [IO.Path]::GetFullPath($env:LVATS_PYTHON)
        if (Test-Path -LiteralPath $configured) { return $configured }
        throw "LVATS_PYTHON 指定的 Python 不存在：$configured"
    }

    $venvPython = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) { return $venvPython }

    $runtime = Join-Path $env:APPDATA 'LobsterAI\runtimes\python-win'
    $preferred = Join-Path $runtime 'Lvats.exe'
    if (Test-Path -LiteralPath $preferred) { return $preferred }
    $fallback = Join-Path $runtime 'python.exe'
    if (Test-Path -LiteralPath $fallback) {
        Write-Host '[Lvats] 警告：未找到 Lvats.exe（带图标），回退运行时 python.exe。'
        return $fallback
    }
    throw "未找到 Python。请先创建 .venv，或用 LVATS_PYTHON 指定带 CUDA PyTorch 的 python.exe。"
}

function Confirm-CudaRuntime([string]$Executable) {
    & $Executable -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 2)" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "GPU 自检失败：$Executable 未能识别 CUDA。"
    }
}

$lock = $null
try {
    try {
        $lockPath = Join-Path $state 'lvats.launch.lock'
        $lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch [IO.IOException] {
        Write-Output '[Lvats] 另一个启动器正在工作，等待服务就绪…'
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            $health = Get-LvatsHealth
            if ($health) {
                if (-not $health.cuda) { throw '已运行的 Lvats 未能识别 GPU，请先双击 stop_lvats.bat 再重新启动。' }
                Write-Output '[Lvats] 服务已就绪（由另一个启动器启动）。'
                exit 0
            }
        }
        throw '另一个启动器未在 30 秒内完成启动。'
    }

    $health = Get-LvatsHealth
    if ($health) {
        if (-not $health.cuda) { throw '已运行的 Lvats 未能识别 GPU，请先双击 stop_lvats.bat 再重新启动。' }
        Write-Output '[Lvats] 服务已在运行，请勿重复启动。'
        exit 0
    }

    if (Test-Path -LiteralPath $pidPath) { Remove-Item -LiteralPath $pidPath -Force }
    $executable = Resolve-LvatsExecutable
    Confirm-CudaRuntime $executable
    Write-Output "[Lvats] GPU 自检通过，正在后台启动 Lvats v1.0.0：$url"
    Write-Output "[Lvats] 进程：$executable"

    $process = Start-Process -FilePath $executable -ArgumentList 'server.py' -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        $health = Get-LvatsHealth
        if ($health) {
            if (-not $health.cuda) {
                throw '新启动的 Lvats 未能识别 GPU，请查看 state\server_8000.out.log。'
            }
            Write-Output '[Lvats] 服务已就绪并在后台运行（无窗口）。日志：state\server_8000.out.log'
            exit 0
        }
        if ($process.HasExited) { throw "Lvats 进程已退出（退出码 $($process.ExitCode)）。" }
    }
    throw '30 秒内未检测到服务就绪，请查看 state\server_8000.err.log。'
} catch {
    Write-Error "[Lvats] $($_.Exception.Message)"
    exit 1
} finally {
    if ($lock) { $lock.Dispose() }
}
