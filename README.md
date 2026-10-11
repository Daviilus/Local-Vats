# Lvats Web

Lvats（Local Video/Audio Transform Service）是运行在 Windows 本机的音视频转录 Web 服务：浏览器交互，Python 后端读取媒体、调用语音模型并生成 TXT、SRT 或 VTT。

当前 Web 版本为 **1.0.1**，原归档基线为 1.0.0。本项目的现行说明与开发范围只涵盖 Web，不提供其他产品的构建或安装指引。

## Features

- 文件/文件夹选择、路径粘贴与网页拖放；文件队列最多 50 项，每页 10 项。
- 本地 Qwen3-ASR 与兼容 faster-whisper 模型；按任务选择、预加载/卸载、闲置释放显存。
- TXT 文本、SRT/VTT 字幕；实时内容、顶部和任务行进度、日志与告警。
- 顺序任务队列、排序、协作取消、持久化恢复、完成任务归档/恢复、结果另存为。
- 快捷提示词、单实例后台启动、本机 HTTPS、页面安全重启和停止服务。
- 独立第三方 API 配置页、可选 HTTP 代理、与本地模型分开的远程音频转录选项。

## Prepare the Python environment

需要 Windows 10/11、Python 3.11、NVIDIA CUDA 环境及加入 PATH 的 FFmpeg/ffprobe。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
# Install a CUDA-compatible PyTorch wheel using the official PyTorch selector.
pip install -r requirements.txt
```

PyTorch 必须是适配显卡/驱动的 CUDA 版本，不能仅按版本号判断 GPU 可用。参考 [PyTorch 安装指南](https://pytorch.org/get-started/locally/)。已有项目 CUDA 运行时可由启动器使用，或通过 `LVATS_PYTHON` 指定解释器绝对路径。

## Start, stop and logs

双击 `启动Lvats.bat`，然后访问 **https://127.0.0.1:8000**。启动脚本使用系统 Windows PowerShell 5.1，初始化本机 HTTPS 证书并后台执行 `server.py`；首次证书初始化会将 Lvats 专用根证书加入当前用户的信任库。

`http://127.0.0.1:8000` 会重定向至同端口 HTTPS。服务只对本机回环开放。此 BAT 链固定使用 HTTPS，不是无证书 HTTP 启动入口。

停止使用页面「停止服务」或 `stop_lvats.bat`；日志使用 `查看Lvats日志.bat`。页面「重启服务」会先保存队列；中断的本地模型任务重启后从头处理，不是逐块断点续传。API 运行中任务被中断后不会自动重新上传，需确认第三方处理/收费状态后手动重建。

## Models and data

代码与运行数据分离。默认数据根为 `%LOCALAPPDATA%\Lvats`：

| Content | Location |
|---|---|
| Models | `data\models` |
| Transcripts | `data\output` |
| Queue | `data\state\queue.json` |
| Preferences / certificates | `config`, `config\ssl` |
| Logs / caches | `logs`, `cache` |
| Process state / temporary uploads | `temp\runtime`, `temp\dropcache` |

源码不含模型权重。首次正式启动会在后台补齐缺失的 Qwen3-ASR 0.6B、1.7B 和字幕配套 Forced Aligner；网页可以在下载时使用。下载失败可在模型管理重试。需要镜像时在启动前设置 `HF_ENDPOINT` 与 `HF_HUB_DISABLE_XET=1`。

内置主模型选项仅为 [Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 与 [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)。Forced Aligner 用于字幕时间轴，不作为主模型。兼容 faster-whisper/CTranslate2 模型可手动放入 `data\models`，再点击「重新扫描」。

Qwen 使用 CUDA，TXT 跳过对齐；SRT/VTT 使用时间轴。faster-whisper GPU 不可用时可降为 CPU，并明确告警。模型切换先释放旧模型，同一时间只保留一个本地 ASR 主模型。

## Transcribe

1. 拖放媒体，或使用文件/文件夹选择、每行一个路径的粘贴输入。
2. 勾选文件，选择模型，按需填写提示词。
3. 点击 TXT、SRT 或 VTT 创建任务。
4. 查看队列与实时内容，完成后下载；原结果保存在 `data\output`。

选择/粘贴路径直接读取原媒体，不复制；普通浏览器拖放无法获得绝对路径，因此经本机回环保存受管理的临时副本。默认本地模型转录不把媒体发送到外部。

支持 MP3、WAV、M4A、FLAC、OGG、AAC、MP4、MOV、MKV、WEBM、AVI、TS、M4V。删除或归档任务记录不删除已生成的输出文件。

## Third-party API and HTTP proxy

1. 点击页头「第三方 API」，进入独立的 `/api-settings` 页面。
2. 输入服务 URL 与模型名称（默认 `whisper-1`），在本机页面输入 API Key。URL 可为 `https://host/v1` 或完整 `.../audio/transcriptions`；公网必须 HTTPS，HTTP 仅供回环模拟/本机服务。
3. 需要代理时点击「API 代理配置」，启用 HTTP 代理，填写 URL（如 `http://127.0.0.1`）及独立端口（如 `7890`），再点击「确定」。关闭或清空代理后同样需要确定保存。
4. 返回首页，在模型下拉框明确选择「API · 模型名称」，勾选文件并选择输出格式；确认音频外传及可能费用后才创建 API 任务。本地模型仍为默认选项，保存配置不会发出第三方请求。

第一版支持 **OpenAI-compatible multipart 音频转录协议**，不是任意聊天/多模态 API；模型和参数支持以服务商为准。只上传抽取的单声道 16kHz/32kbps MP3，每块最多 10 分钟且小于 24MiB，不发送原视频、源文件路径或元数据。跨块语境可能影响识别；需要本机 FFmpeg 支持 `libmp3lame` 编码。

TXT 使用 JSON 文本；SRT/VTT 要求 `verbose_json` 和完整有效的分段时间轴。服务不支持字幕、分段缺文本或时间轴无效时任务失败，不伪造字幕，可改选 TXT。累计结果上限为 1MiB 文本、20,000 段，实时预览最多保留末尾 20,000 字符，输出文件保留上限内完整结果。

安全配置说明：

- Key 由 Windows 当前用户 DPAPI 加密保存于 `config\remote-api.json`；GET 只返回“已设置”状态，不返回密钥。留空保留已有 Key，勾选清除后需确认；其他用户无法直接解密复制过来的配置。不要在聊天或 URL 中填写 Key。
- 代理仅支持无认证 HTTP，URL 不带端口/路径/用户名密码，端口为 1～65535。设置用于后端实际调用；不默用环境代理。访问 HTTPS API 时通过 HTTP CONNECT 保持 TLS 校验；回环 HTTP API 只允许回环代理。
- 保存/应用不等于账号、服务、代理已通过连通性测试。服务URL留空可禁用远程配置，但不会自动清除已存密钥。
- 配置页和任务绑定修订号；旧标签页或排队任务遇配置变化会拒绝继续，需重读并确认。认证失败、限流、超时、代理不可达及重定向安全失败，不自动付费重试，也不静默改用本地模型。
- 取消停止本机请求，但不能撤回第三方已接收的音频或费用。受控停服给予请求/FFmpeg清理有界窗口；断电、外部强杀或清理超时仍可能留下当前临时音频，应只在确认无活动任务后人工检查数据根 `temp` 内本次目录，不能盲删用户数据。

从旧 1.0.0 更新后，需在任务安全的时机手动重启 Lvats 服务并刷新网页，后端新路由才会加载。正在运行的旧服务不会因文件修改自动升级；如果页面提示“当前服务尚未加载此功能”，请使用新版服务。本轮开发没有替您停止或重启 8000。

## Development and verification

仓库只发布 Web 源码、启动辅助脚本和测试；不包含模型、用户数据、证书、API 配置或内部开发记录。

使用明确的 Python 3.11 解释器，测试不加载真实模型。单元检查使用占位模式和独立数据根；路径测试验证默认值时要清除 `ASR_DATA_ROOT` 覆盖。

```powershell
$python = ".\.venv\Scripts\python.exe"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:ASR_TEST_MODE = "1"
$env:ASR_PLACEHOLDER = "1"
$env:ASR_PORT = "8001"
$env:ASR_DATA_ROOT = Join-Path $PWD (".cowork-temp\web-test-" + [guid]::NewGuid().ToString("N"))
& $python -m unittest discover -s tests -p "test_remote*.py" -v
& $python -m unittest discover -s tests -p "test_regressions.py" -v
& $python -m unittest discover -s tests -p "test_version.py" -v
$env:ASR_DATA_ROOT = $null
& $python -m unittest discover -s tests -p "test_paths.py" -v
& $python -m unittest discover -s tests -p "test_windows_launcher.py" -v
```

浏览器冒烟额外需要 `playwright` 与本机 Microsoft Edge；使用 FFmpeg/ffprobe 抽取合成音频，不调用付费 API。

```powershell
& $python -m pip install playwright
& $python tests\remote_api_smoke.py
& $python tests\integration_smoke.py
```

默认使用 8001，确认其空闲且未被系统保留后再执行，两项冒烟必须依次运行。Windows 将 8001 列入保留端口段时，可先设置 `$env:ASR_SMOKE_PORT = "18001"` 改用已确认空闲的测试端口；8000 明确禁止作为该覆盖值。测试使用回环模拟 API、假密钥和独立目录，不写入、停止或重启正式 8000；API 冒烟只读访问其根页面以比较测试前后的可访问状态。`https_redirect_smoke.py` 和 `real_qwen_smoke.py` 是显式验证脚本：分别需要事先准备的本机证书 fixture、短音频及模型；不要未经审阅直接运行或为真实推理停掉正在使用的服务。
