# Lvats

Lvats（Local Video/Audio Transform Service）是一款在 Windows 本机运行的音视频转录工具。它提供浏览器界面，可批量添加音频或视频、选择识别模型并生成 TXT、SRT 或 VTT 文件。

主要功能：

- 文件、文件夹、路径粘贴和网页拖放导入
- 最多 50 个文件的转录队列，以及排序、取消、归档和恢复
- TXT 纯文本转录，SRT/VTT 字幕与时间轴生成
- 快捷提示词、实时进度、运行日志和任务持久化
- 模型按需加载、切换、预加载和闲置自动卸载
- 所有文件、模型和转录结果均保留在本机

## 使用方法

### 1. 准备环境

需要：

- Windows 10 或 Windows 11
- Python 3.11
- NVIDIA 显卡及可用的 CUDA 环境
- FFmpeg，并已加入 `PATH`

创建虚拟环境并安装依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

请根据自己的 CUDA 版本，从 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 安装对应的 CUDA 版 PyTorch。

### 2. 首次启动与模型下载

双击 `启动Lvats.bat`。首次启动会为本机 HTTPS 创建并安装 Lvats 专用证书，然后在后台启动服务。浏览器访问：

```text
https://127.0.0.1:8000
```

模型权重不包含在 GitHub 源码中。首次正式启动后，Lvats 会在后台自动从 Hugging Face 下载 Qwen3-ASR 0.6B、1.7B 和字幕时间轴所需的 Qwen3 Forced Aligner，并保存到 `models/`。下载过程中界面可以正常打开；模型完成下载并注册后即可创建转录任务。请预留足够的磁盘空间。

下载失败时可在“模型管理”中重试。如果所在网络需要 Hugging Face 镜像，请在启动前设置 `HF_ENDPOINT` 和 `HF_HUB_DISABLE_XET=1`。

如需停止服务，可点击页面右上角的“停止服务”，或双击 `stop_lvats.bat`。运行日志可通过 `查看Lvats日志.bat` 查看。

### 3. 转录

1. 拖放文件，或使用“选择文件”“选择文件夹”和路径粘贴添加音视频。
2. 勾选要处理的文件，选择模型，并按需填写提示词。
3. 点击 TXT、SRT 或 VTT 创建任务。
4. 在任务列表查看进度；完成后下载结果，文件也会保存在 `output/`。

支持的输入格式：MP3、WAV、M4A、FLAC、OGG、AAC、MP4、MOV、MKV、WEBM、AVI、TS、M4V。

## 内置模型列表

Lvats 1.0.0 的内置模型选项仅包括：

- [`Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)：占用较低，适合作为默认模型
- [`Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)：占用较高，适合优先考虑识别效果的场景

“内置”指程序原生识别并展示这些模型；模型权重仍需用户自行下载。

## 支持的模型类型

- **Qwen3-ASR（Hugging Face Transformers 格式）**：支持 0.6B 和 1.7B；使用 CUDA 推理。SRT/VTT 可配合 Qwen3 Forced Aligner 生成时间轴。
- **faster-whisper（CTranslate2 格式）**：可将兼容模型目录放入 `models/` 后重新扫描；优先使用 GPU，必要时可降级到 CPU int8。

Lvats 会识别 `models/` 下的兼容模型目录。切换模型时会先释放旧模型，同一时间只保留一个 ASR 主模型。
