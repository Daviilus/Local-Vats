# -*- coding: utf-8 -*-
"""
asr_model.py — Qwen3-ASR 模型推理模块（v0.5.4）

v0.5.4 变更（老板拍板）：
  - ASR 主模型改纯 fp16（H2 实测 int8 核慢 ≈4.7 倍）；对齐器改 int8 量化补省显存
  - ASR_LOAD_IN_8BIT 默认 0；新增 ASR_ALIGNER_IN_8BIT（默认 1）；量化失败自动退回 fp16

v0.5.0 变更：
  - WavCache：按【文件】缓存 16k 单声道 wav（不是按任务），同文件多任务复用，
    该文件最后一个任务结束后删除缓存；切块只从缓存 wav 切，不再回头碰源视频
  - 解码显式贪心 num_beams=1 + max_new_tokens 上限；输出触顶即告警"疑似截断"
  - device_map 显式 "cuda"；加载后检测是否被 offload 到 CPU，禁止静默降级
  - 每切块三段耗时日志：[chunk i/n] 抽取 x.xs | 识别 x.xs | 后处理 x.xs
  - 协作式取消：cancel token 在切块边界检查，命中即停（延迟 ≤ 一个切块）
  - need_timestamps=False（txt 任务）跳过 ForcedAligner，省下逐块对齐开销
  - 修复致命隐患：对齐 OOM 不再卸载 ASR 主模型（否则之后每一块都会重新加载+量化）

资源占用约定：
  - 服务启动时【不加载】模型（见 server.py 的 lifespan）
  - 第一次调用 transcribe_audio() 时才加载 ASR 模型
  - 转录完成后默认保留在显存；闲置自动卸载 / 手动 /api/unload

模型分工：
  - Qwen3-ASR-1.7B-hf          ：语音识别（文本）
  - Qwen3-ForcedAligner-0.6B-hf：强制对齐（逐词时间戳 → 合成 segments）
    ASR 主模型本身不输出时间戳，官方方案就是用 ForcedAligner 补时间戳。
    仅 srt / vtt 需要；txt 任务直接跳过。

GPU 约定：强制只用 GPU（CUDA only）。没有可用 CUDA 时直接报错，不回退 CPU。
音频预处理：需要 ffmpeg（统一转 16kHz 单声道 wav，视频同时提取音轨）。
"""

from __future__ import annotations

import gc
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 模型目录名容错（兼容大小写/写法差异）
MODEL_DIR_CANDIDATES = [
    "Qwen3-ASR-1.7B-hf",
    "qwen-3-asr-1.7B-Hf",
    "qwen3-asr-1.7b-hf",
]
ALIGNER_DIR_CANDIDATES = [
    os.environ.get("ASR_ALIGNER_DIR", ""),
    "Qwen3-ForcedAligner-0.6B-hf",
    "models/Qwen3-ForcedAligner-0.6B-hf",
    "qwen3-forcedaligner-0.6b-hf",
]

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

# ──────────────────────────────────────────────────────────────
# 可调参数（全部支持环境变量覆盖，改完需重启生效）
# ──────────────────────────────────────────────────────────────
# 单次推理最多处理的音频秒数（防爆显存关键参数）
CHUNK_SECONDS = int(os.environ.get("ASR_CHUNK_SECONDS", "60"))
# 解码束宽度：1 = 贪心（最快）。调大更准但更慢。
NUM_BEAMS = int(os.environ.get("ASR_NUM_BEAMS", "1"))
# 每块最多生成 token 数。触顶会告警"疑似截断"，可调大。
MAX_NEW_TOKENS = int(os.environ.get("ASR_MAX_NEW_TOKENS", "1024"))
# 显式指定设备，禁止 device_map="auto" 把层偷偷 offload 到 CPU
DEVICE_MAP = (os.environ.get("ASR_DEVICE_MAP", "cuda") or "cuda").strip()
# 是否用 bitsandbytes 做 int8 权重量化（ASR 主模型）。
# v0.5.3 及之前默认 1（int8）；H2 实测 int8 核比 fp16 慢 ≈4.7 倍（54.3s/块 → 11.5s/块），
# 2026-09-03 老板拍板：ASR 主模型改纯 fp16（显存 ≈4.9GB，见 DEBUG_REPORT.md §9），
# 对齐器改 int8 补省显存。如需回退旧行为：设 ASR_LOAD_IN_8BIT=1 + ASR_ALIGNER_IN_8BIT=0。
LOAD_IN_8BIT = os.environ.get("ASR_LOAD_IN_8BIT", "0").strip().lower() in ("1", "true", "yes")
# 对齐器是否 int8 量化（0.6B 小模型，int8 ≈ 省一半显存；与 fp16 主模型搭配用）。
ALIGNER_IN_8BIT = os.environ.get("ASR_ALIGNER_IN_8BIT", "1").strip().lower() in ("1", "true", "yes")
# 【预留】当前实现直接调 model.generate（单条 60s 音频，无 batch 维度），
# 该值暂不生效，仅在 /api/health 中回显，供日后改 pipeline 时使用。
BATCH_SIZE = int(os.environ.get("ASR_BATCH_SIZE", "4"))
# wav 缓存目录与上限（MB）
WAV_CACHE_MAX_MB = float(os.environ.get("ASR_WAV_CACHE_MAX_MB", "4096"))
TEST_MODE = os.environ.get("ASR_TEST_MODE", "").strip().lower() in ("1", "true", "yes")
CACHE_DIR = Path(os.environ.get("ASR_WAV_CACHE_DIR", "") or
                 (BASE_DIR / (".wavcache.test" if TEST_MODE else ".wavcache")))

# 模型闲置多久后自动卸载（秒），默认 5 分钟
IDLE_TIMEOUT_SECONDS = float(os.environ.get("ASR_IDLE_TIMEOUT", "300"))

# 占位模式：ASR_PLACEHOLDER=1 时不加载模型，返回固定占位结果。
# 用途：自检 / 联调流程（不占显存，绝不与正式服务争抢）。
USE_PLACEHOLDER = os.environ.get("ASR_PLACEHOLDER", "").strip().lower() in ("1", "true", "yes")

# 懒加载全局状态（线程安全）
_lock = threading.RLock()
INFERENCE_LOCK = threading.RLock()   # 两引擎推理、预加载和卸载共享此锁
_processor = None
_model = None
_model_path = None
_device = None
_aligner_processor = None
_aligner_model = None
_quantized = False
_aligner_quantized = False         # 对齐器是否成功走了 int8（独立于主模型）
_device_info: dict = {}
_align_disabled = False             # 对齐反复失败后关闭后续对齐，避免拖慢主流程

# idle 自动卸载状态
_last_used_at = time.time()
_idle_timer = None
_idle_unload_active = False


def _norm_path(p) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


# ──────────────────────────────────────────────────────────────
# wav 缓存（按文件，不是按任务）
# ──────────────────────────────────────────────────────────────
class WavCache:
    """
    按【源文件】缓存 16k 单声道 wav。

    - 键：规范化路径 sha1[:16]；另用 (mtime_ns, size) 做失效校验
    - 引用计数按 task_id：同一文件的多个任务（txt/srt/vtt）共用一份缓存
    - release(keep=False) 在无任何引用时才真正删除文件
    - 超过 WAV_CACHE_MAX_MB 时 LRU 淘汰（只淘汰无引用的条目）
    """

    def __init__(self, directory: Path, max_mb: float):
        self._dir = directory
        self._max_bytes = max_mb * 1e6
        self._lock = threading.Lock()
        self._rec: dict[str, dict] = {}     # key -> {wav, mtime_ns, size, last, bytes}
        self._peers: dict[str, set] = {}    # key -> {task_id, ...}

    # ── 内部 ──
    @staticmethod
    def _key(src_norm: str) -> str:
        return hashlib.sha1(src_norm.encode("utf-8")).hexdigest()[:16]

    def _fresh(self, key: str, src: Path) -> bool:
        rec = self._rec.get(key)
        if not rec:
            return False
        try:
            st = os.stat(src)
        except OSError:
            return False
        return (rec["mtime_ns"] == st.st_mtime_ns and rec["size"] == st.st_size
                and rec["wav"].exists())

    def _evict_if_needed(self):
        total = sum(r["bytes"] for r in self._rec.values())
        if total <= self._max_bytes:
            return
        # 只淘汰无引用的，按 last 升序
        victims = sorted((r for k, r in self._rec.items() if not self._peers.get(k)),
                         key=lambda r: r["last"])
        for r in victims:
            if total <= self._max_bytes:
                break
            try:
                r["wav"].unlink(missing_ok=True)
            except Exception:
                pass
            total -= r["bytes"]
            for k, v in list(self._rec.items()):
                if v is r:
                    self._rec.pop(k, None)
                    self._peers.pop(k, None)

    # ── 对外 ──
    def acquire(self, src: Path, task_id: str) -> tuple[Path, bool]:
        """返回 (wav 路径, 是否缓存命中)。未命中则执行唯一一次真实抽取。"""
        src = Path(src)
        key = self._key(_norm_path(src))
        with self._lock:
            if self._fresh(key, src):
                self._peers.setdefault(key, set()).add(task_id)
                self._rec[key]["last"] = time.time()
                return self._rec[key]["wav"], True

            self._dir.mkdir(parents=True, exist_ok=True)
            wav = self._dir / f"{key}.wav"
            st = os.stat(src)
            _ffmpeg_extract(src, wav)
            self._rec[key] = {"wav": wav, "mtime_ns": st.st_mtime_ns, "size": st.st_size,
                              "last": time.time(), "bytes": wav.stat().st_size}
            self._peers[key] = {task_id}
            self._evict_if_needed()
            return wav, False

    def release(self, src: Path, task_id: str, keep: bool = False):
        """
        释放本任务对缓存的引用。
        keep=True  → 同文件还有其它任务在用，只解除引用，不删文件。
        keep=False → 本任务已是最后一个，删除缓存文件。
        """
        key = self._key(_norm_path(src))
        with self._lock:
            self._peers.get(key, set()).discard(task_id)
            if keep or self._peers.get(key):
                return
            rec = self._rec.pop(key, None)
            self._peers.pop(key, None)
            if rec:
                try:
                    rec["wav"].unlink(missing_ok=True)
                except Exception:
                    pass

    def drop_all(self):
        """启动时清空：绝不留脏数据。"""
        with self._lock:
            self._rec.clear()
            self._peers.clear()
        shutil.rmtree(self._dir, ignore_errors=True)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._rec),
                    "mb": round(sum(r["bytes"] for r in self._rec.values()) / 1e6, 1)}


wav_cache = WavCache(CACHE_DIR, WAV_CACHE_MAX_MB)


# ──────────────────────────────────────────────────────────────
# 目录与加载
# ──────────────────────────────────────────────────────────────
def _find_dir(candidates: list[str]) -> Path | None:
    for name in candidates:
        if not name:
            continue
        p = Path(name) if Path(name).is_absolute() else BASE_DIR / name
        if p.exists():
            return p
    return None


def find_aligner_dir() -> Path | None:
    """只接受配置与权重都完整的对齐模型目录，避免把中断下载当成可用模型。"""
    for name in ALIGNER_DIR_CANDIDATES:
        if not name:
            continue
        path = Path(name) if Path(name).is_absolute() else BASE_DIR / name
        has_weights = ((path / "model.safetensors").is_file()
                       or (path / "model.safetensors.index.json").is_file()
                       or (path / "pytorch_model.bin").is_file())
        if (path / "config.json").is_file() and has_weights:
            return path
    return None


def find_model_dir(model_dir=None) -> Path:
    if model_dir is not None:
        p = Path(model_dir).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"所选 ASR 模型目录不存在：{p}")
        return p
    p = _find_dir(MODEL_DIR_CANDIDATES)
    if p is None:
        raise FileNotFoundError(f"找不到 ASR 模型目录，期望之一: {MODEL_DIR_CANDIDATES}")
    return p


def _load_torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("缺少 torch，请先执行: pip install -r requirements.txt") from e
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA GPU：本应用被配置为仅使用 GPU 推理。")
    return torch


def _load_half_precision(from_pretrained, model_dir: str):
    """纯 float16 半精度加载（对齐模型等小模型用）。"""
    torch = _load_torch()
    kwargs = dict(device_map=DEVICE_MAP, low_cpu_mem_usage=True)
    try:
        return from_pretrained(model_dir, torch_dtype=torch.float16, **kwargs)
    except TypeError:
        return from_pretrained(model_dir, dtype=torch.float16, **kwargs)


def _load_quantized(from_pretrained, model_dir: str):
    """
    ASR 主模型加载：int8 权重量化（BitsAndBytesConfig）+ float16 计算。
    transformers 5.x 已移除 from_pretrained(load_in_8bit=...) 直传参数，
    必须走 quantization_config=...，否则报 unexpected keyword argument。
    量化失败自动退回纯 float16，保证服务可用。
    device_map 显式取 DEVICE_MAP（默认 "cuda"），不用 "auto" —— auto 会在显存
    不够时把部分层静默 offload 到 CPU，表现为"没报错但极慢"。
    """
    global _quantized
    torch = _load_torch()
    base = dict(device_map=DEVICE_MAP, low_cpu_mem_usage=True)

    def _call(**kw):
        try:
            return from_pretrained(model_dir, torch_dtype=torch.float16, **kw)
        except TypeError:
            return from_pretrained(model_dir, dtype=torch.float16, **kw)

    quant = None
    if LOAD_IN_8BIT:
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            quant = None

    if quant is not None:
        try:
            m = _call(quantization_config=quant, **base)
            _quantized = True
            return m
        except Exception as e:
            print(f"[asr_model] int8 量化加载失败（{e}），退回 float16 半精度", flush=True)
    _quantized = False
    return _call(**base)


def _compute_device_info():
    """加载后立刻算出实际设备分布，用于检测"静默 CPU 降级"。"""
    global _device_info
    info: dict = {"mode": "real", "placeholder": False,
                  "requested_device_map": DEVICE_MAP,
                  "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS,
                  "chunk_seconds": CHUNK_SECONDS, "load_in_8bit": LOAD_IN_8BIT,
                  "batch_size": BATCH_SIZE, "batch_size_active": False}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda_available"] = False
        info["gpu_name"] = None

    dm = getattr(_model, "hf_device_map", None) or {}
    cpu_layers = [str(k) for k, v in dm.items() if str(v) == "cpu"]
    info.update({
        "engine": "qwen", "model_path": _model_path,
        "device": str(_device) if _device is not None else None,
        "quantized": bool(_quantized),
        "dtype": str(getattr(_model, "dtype", "") or ""),
        "cpu_layers": cpu_layers,
        "degraded": bool(cpu_layers),
        "aligner_loaded": _aligner_model is not None,
        "aligner_quantized": bool(_aligner_quantized),
        "aligner_in_8bit_config": ALIGNER_IN_8BIT,
    })
    _device_info = info
    if info["degraded"]:
        print(f"[asr_model] ⚠ CPU 降级：以下层被 offload 到 CPU —— {cpu_layers}", flush=True)
    return info


def get_device_info() -> dict:
    if USE_PLACEHOLDER:
        info = {"mode": "placeholder", "placeholder": True,
                "requested_device_map": DEVICE_MAP,
                "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS,
                "chunk_seconds": CHUNK_SECONDS, "load_in_8bit": LOAD_IN_8BIT,
                "batch_size": BATCH_SIZE, "batch_size_active": False,
                "cuda_available": None, "gpu_name": None,
                "device": None, "quantized": False, "dtype": "",
                "cpu_layers": [], "degraded": False}
        return info
    if not _device_info and _model is not None:
        _compute_device_info()
    if _device_info:
        return {**_device_info, "aligner_loaded": _aligner_model is not None,
                "aligner_quantized": bool(_aligner_quantized) if _aligner_model is not None else False}
    return dict(_device_info) if _device_info else {
        "mode": "unloaded", "placeholder": False, "requested_device_map": DEVICE_MAP,
        "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS,
        "chunk_seconds": CHUNK_SECONDS, "load_in_8bit": LOAD_IN_8BIT,
        "batch_size": BATCH_SIZE, "batch_size_active": False,
        "degraded": False, "cpu_layers": [],
    }


def load_model(model_dir=None):
    """按规范化路径复用；换模型时先释放旧权重，避免双模型显存峰值。"""
    with INFERENCE_LOCK, _lock:
        path = str(find_model_dir(model_dir))
        if _model is not None and _model_path == _norm_path(path):
            touch()
            return _processor, _model
        unload_model()
        return _load_model(path)


def _load_model(model_dir):
    """懒加载 ASR 主模型（第一次转录时调用）。强制 GPU + fp16/int8 + 显式 device_map。"""
    global _processor, _model, _device, _model_path

    _load_torch()

    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    processor = AutoProcessor.from_pretrained(model_dir)
    try:
        model = _load_quantized(AutoModelForSpeechSeq2Seq.from_pretrained, model_dir)
    except Exception as e:
        # 兜底：个别 transformers 版本下 AutoModelForSpeechSeq2Seq
        # 未注册该架构，则退回官方 README 推荐的 AutoModelForMultimodalLM
        print(f"[asr_model] AutoModelForSpeechSeq2Seq 加载失败（{e}），改用 AutoModelForMultimodalLM",
              flush=True)
        from transformers import AutoModelForMultimodalLM
        model = _load_quantized(AutoModelForMultimodalLM.from_pretrained, model_dir)

    model.eval()

    _processor = processor
    _model = model
    _model_path = _norm_path(model_dir)
    _device = model.device if hasattr(model, "device") else "cuda"
    _compute_device_info()
    touch()
    print(f"[asr_model] 已加载所选模型：{model_dir}", flush=True)
    return processor, model


def load_aligner():
    """懒加载强制对齐模型；本地目录不存在时返回 None。

    v0.5.4：默认 int8 量化（ASR_ALIGNER_IN_8BIT=1）+ float16 计算，
    给 fp16 主模型省显存；量化失败自动退回 float16，保证可用。
    """
    global _aligner_processor, _aligner_model, _aligner_quantized

    if _aligner_model is not None:
        return _aligner_processor, _aligner_model

    aligner_dir = find_aligner_dir()
    if aligner_dir is None:
        return None, None

    _load_torch()
    from transformers import AutoProcessor, AutoModelForTokenClassification

    _aligner_processor = AutoProcessor.from_pretrained(str(aligner_dir))
    torch = _load_torch()
    base = dict(device_map=DEVICE_MAP, low_cpu_mem_usage=True)

    def _call(**kw):
        try:
            return AutoModelForTokenClassification.from_pretrained(str(aligner_dir), torch_dtype=torch.float16, **kw)
        except TypeError:
            return AutoModelForTokenClassification.from_pretrained(str(aligner_dir), dtype=torch.float16, **kw)

    _aligner_quantized = False
    model = None
    if ALIGNER_IN_8BIT:
        try:
            from transformers import BitsAndBytesConfig
            model = _call(quantization_config=BitsAndBytesConfig(load_in_8bit=True), **base)
            _aligner_quantized = True
        except Exception as e:
            print(f"[asr_model] 对齐器 int8 量化加载失败（{e}），退回 float16", flush=True)
            model = None
    if model is None:
        model = _call(**base)

    _aligner_model = model
    _aligner_model.eval()
    print(f"[asr_model] 对齐器已加载（{'int8' if _aligner_quantized else 'fp16'}）", flush=True)
    return _aligner_processor, _aligner_model


def unload_model() -> None:
    """卸载全部模型，释放显存（对应前端「卸载模型」按钮 / idle 自动卸载）。"""
    global _processor, _model, _device, _aligner_processor, _aligner_model, _device_info
    global _model_path, _align_disabled
    with INFERENCE_LOCK, _lock:
        _model_path = None
        _align_disabled = False
        if _model is None and _aligner_model is None:
            return
        _model = None
        _processor = None
        _aligner_model = None
        _aligner_processor = None
        _device = None
        _device_info = {}
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def touch() -> None:
    """记录一次模型使用时间，用于 idle 自动卸载。"""
    global _last_used_at
    _last_used_at = time.time()


def preload(model_dir=None) -> None:
    """后台预加载模型（线程安全、幂等）。"""
    if USE_PLACEHOLDER:
        return
    with INFERENCE_LOCK:
        load_model(model_dir)
        touch()


def seconds_since_last_use() -> float:
    return time.time() - _last_used_at


def _idle_unload_loop() -> None:
    """后台线程：模型闲置超过 IDLE_TIMEOUT_SECONDS 后自动卸载。"""
    while _idle_unload_active:
        time.sleep(15)
        if not is_loaded():
            continue
        if INFERENCE_LOCK.acquire(blocking=False):
            try:
                if seconds_since_last_use() >= IDLE_TIMEOUT_SECONDS:
                    print(f"[asr_model] 模型闲置 {IDLE_TIMEOUT_SECONDS:.0f} 秒，自动卸载以释放显存", flush=True)
                    unload_model()
            finally:
                INFERENCE_LOCK.release()


def start_idle_watcher() -> None:
    """启动 idle 自动卸载守护线程（应在服务启动时调用一次）。"""
    global _idle_timer, _idle_unload_active
    if _idle_unload_active:
        return
    _idle_unload_active = True
    _idle_timer = threading.Thread(target=_idle_unload_loop, daemon=True, name="asr-idle-watcher")
    _idle_timer.start()


def is_loaded() -> bool:
    return _model is not None


def aligner_available() -> bool:
    return find_aligner_dir() is not None


# ──────────────────────────────────────────────────────────────
# ffmpeg 音频工具
# ──────────────────────────────────────────────────────────────
def _ffmpeg_extract(src: Path, dst: Path) -> None:
    """用 ffmpeg 一次性把任意音频/视频转成 16kHz 单声道 wav。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-nostdin", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 转码失败：{proc.stderr.strip()[-500:]}")


def _duration(path: Path) -> float:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def _split_wav(wav: Path, seconds: int) -> list[Path]:
    """用 ffmpeg segment 把 wav 切成 <=seconds 秒的块。"""
    out_dir = Path(tempfile.mkdtemp(prefix="asr_chunks_"))
    cmd = [
        FFMPEG, "-y", "-nostdin", "-loglevel", "error",
        "-i", str(wav),
        "-f", "segment", "-segment_time", str(seconds),
        "-c", "copy",
        str(out_dir / "chunk_%05d.wav"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 分块失败：{proc.stderr.strip()[-500:]}")
    return sorted(out_dir.glob("chunk_*.wav"))


# ──────────────────────────────────────────────────────────────
# 时间戳 → segments
# ──────────────────────────────────────────────────────────────
_PUNCT_HARD = set("。！？!?；;…")       # 硬断句：立即收尾
_MAX_CHARS = 22                          # 单条字幕最大字数
_MAX_DUR = 6.0                           # 单条字幕最大时长（秒）
_MAX_GAP = 0.8                           # 词间停顿超过该值则分段


def merge_words_to_segments(words: list[dict]) -> list[dict]:
    """把逐词时间戳合并成字幕级 segments。"""
    segments: list[dict] = []
    cur_text = ""
    cur_start = None
    cur_end = 0.0

    def flush():
        nonlocal cur_text, cur_start
        if cur_text.strip() and cur_start is not None:
            seg_start = round(cur_start, 3)
            seg_end = round(cur_end, 3)
            if seg_end <= seg_start:
                # 零长度块保护：对齐器单字词 start==end 时给最小可见时长 40ms
                seg_end = round(seg_start + 0.04, 3)
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": cur_text.strip(),
            })
        cur_text = ""
        cur_start = None

    for w in words:
        if cur_start is None:
            cur_start = w["start_time"]
        elif w["start_time"] - cur_end > _MAX_GAP:
            flush()
            cur_start = w["start_time"]

        cur_text += w["text"]
        cur_end = w["end_time"]

        ends_punct = bool(w["text"]) and w["text"][-1] in _PUNCT_HARD
        if ends_punct or len(cur_text) >= _MAX_CHARS or (cur_end - cur_start) >= _MAX_DUR:
            flush()

    flush()
    return segments


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────
def _placeholder_result() -> dict:
    """占位结果（仅用于联调流程，非真实转录）。"""
    return {
        "text": "这是占位转录结果。请把 ASR_PLACEHOLDER 关闭以启用真实模型推理。",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "这是占位转录结果。"},
            {"start": 2.5, "end": 5.0, "text": "请关闭 ASR_PLACEHOLDER 以启用真实模型推理。"},
        ],
    }


def transcribe_audio(audio_path, prompt: str | None = None, progress_cb=None,
                     cancel=None, need_timestamps: bool = True, job_id: str = "",
                     *, model_dir=None) -> dict:
    """
    转录入口（模型接口约定）：

        transcribe_audio(audio_path, prompt=None, progress_cb=None,
                         cancel=None, need_timestamps=True, job_id="") -> {
            "text": "完整文本",
            "segments": [{"start": 0.0, "end": 2.5, "text": "一句话"}, ...],
            "cancelled": False,
        }

    prompt          ：提示词/热词，作为 system prompt 引导转录（可选）
    progress_cb     ：progress_cb(percent, message, partial=None, meta=None)
    cancel          ：协作式取消令牌，需提供 .is_set() -> bool
    need_timestamps ：False 时跳过 ForcedAligner（txt 任务不需要时间戳，可省掉逐块对齐）
    job_id          ：任务 id，用于 wav 缓存的引用计数
    """
    if USE_PLACEHOLDER:
        return _transcribe_placeholder(prompt, progress_cb, cancel, need_timestamps)

    audio_path = Path(audio_path)
    with INFERENCE_LOCK:
        try:
            return _transcribe_locked(audio_path, prompt, progress_cb, cancel,
                                      need_timestamps, job_id, model_dir=model_dir)
        finally:
            touch()


def _cancelled(cancel) -> bool:
    try:
        return bool(cancel is not None and cancel.is_set())
    except Exception:
        return False


def _transcribe_placeholder(prompt, progress_cb, cancel, need_timestamps) -> dict:
    """占位模式：不加载模型，但要完整走一遍 分块/进度/取消/耗时日志，便于自检。"""
    import os as _os
    slow = _os.environ.get("ASR_PLACEHOLDER_SLOW", "").strip().lower() in ("1", "true", "yes")
    steps = 8 if slow else 3

    def report(pct, msg, partial=None, meta=None):
        if progress_cb:
            try:
                progress_cb(pct, msg, partial, meta)
            except TypeError:
                progress_cb(pct, msg)
            except Exception:
                pass

    report(5.0, "占位模式：音频预处理中", None,
           {"timing": {"prep_s": 0.1, "cache_hit": False, "kind": "prep"}})
    texts = []
    for i in range(1, steps + 1):
        if _cancelled(cancel):
            report(100.0, "占位模式：已取消", "".join(texts),
                   {"timing": {"kind": "cancelled", "chunk": i, "total": steps}})
            return {"text": "".join(texts), "segments": [], "cancelled": True}
        t0 = time.perf_counter()
        if slow:
            time.sleep(0.6)
        seg = f"（占位第 {i}/{steps} 块）这是占位转录文本，用于验证分块进度、SSE 推送与取消链路。"
        texts.append(seg)
        t_rec = time.perf_counter() - t0
        pct = round(5.0 + i / steps * 90.0, 1)
        report(pct, f"占位模式：已识别第 {i}/{steps} 块", "".join(texts),
               {"timing": {"kind": "chunk", "chunk": i, "total": steps,
                           "extract_s": 0.0, "recog_s": round(t_rec, 2), "post_s": 0.0,
                           "cached": True}})
    report(100.0, "占位模式完成", "".join(texts), {"timing": {"kind": "done"}})
    res = _placeholder_result()
    res["text"] = "".join(texts)
    if not need_timestamps:
        res["segments"] = []
    res["cancelled"] = False
    return res


def _transcribe_one_chunk(wav: Path, prompt: str | None) -> tuple[str, str | None, bool]:
    """对单个 <=CHUNK_SECONDS 的 wav 做 ASR 推理，返回 (text, language, 是否疑似截断)。"""
    import torch

    processor, model = _processor, _model
    inputs = processor.apply_transcription_request(audio=str(wav), prompt=prompt or None)
    inputs = inputs.to(_device, torch.float16)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                    num_beams=NUM_BEAMS, do_sample=False)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    truncated = generated_ids.shape[-1] >= MAX_NEW_TOKENS

    try:
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
        text = (parsed.get("transcription") or "").strip()
        language = parsed.get("language")
    except Exception:
        raw = processor.decode(generated_ids, skip_special_tokens=False)[0]
        marker = "<asr_text>"
        text = (raw.split(marker, 1)[1] if marker in raw else raw).strip()
        language = None
    return text, language, truncated


def _transcribe_locked(audio_path: Path, prompt: str | None = None, progress_cb=None,
                       cancel=None, need_timestamps: bool = True, job_id: str = "",
                       *, model_dir=None) -> dict:
    global _processor, _model, _device, _align_disabled

    def report(pct: float, msg: str, partial: str | None = None, meta: dict | None = None):
        if progress_cb:
            try:
                progress_cb(pct, msg, partial, meta)
            except TypeError:
                # 兼容只接受两个参数的回调
                progress_cb(pct, msg)
            except Exception:
                pass

    # 1) 懒加载：第一次才加载模型
    touch()
    _align_disabled = False  # OOM 禁用仅限本次任务
    if _model is None or (model_dir is not None and
                         _model_path != _norm_path(find_model_dir(model_dir))):
        if _cancelled(cancel):
            return {"text": "", "segments": [], "cancelled": True}
        report(2.0, "加载模型中（首次较慢，请耐心等待）")
        load_model(model_dir)
        touch()
    if _cancelled(cancel):
        return {"text": "", "segments": [], "cancelled": True}

    # 模型就绪即上报设备实况（禁止静默 CPU 降级：一旦有层被 offload 就明牌告警）
    _info = get_device_info()
    report(4.0, f"模型已加载（{_info.get('device')} / "
                f"{'int8' if _info.get('quantized') else 'fp16'}）", None,
           {"device": _info, "level": "t",
            "log": f"device={_info.get('device')} quantized={_info.get('quantized')} "
                   f"dtype={_info.get('dtype')} degraded={_info.get('degraded')}"})
    if _info.get("degraded"):
        report(4.0, "⚠ 检测到 CPU 降级", None,
               {"level": "warn", "device": _info,
                "log": f"⚠ CPU 降级：以下层被放到 CPU —— {_info.get('cpu_layers')}"})

    report(5.0, "音频预处理中")

    # 2) 按【文件】缓存 16k 单声道 wav（同文件多任务复用，只抽一次）
    t_prep = time.perf_counter()
    wav, hit = wav_cache.acquire(audio_path, job_id or f"thread-{threading.get_ident()}")
    prep_s = time.perf_counter() - t_prep
    report(6.0, f"音频预处理完成（{'缓存命中' if hit else '新建'} {prep_s:.1f}s）", None,
           {"timing": {"kind": "prep", "prep_s": round(prep_s, 2), "cache_hit": hit,
                       "source": str(audio_path)}})

    chunks_dir: Path | None = None
    try:
        # 3) 从缓存 wav 切块（不再回头碰源视频）
        duration = _duration(wav)
        if duration > CHUNK_SECONDS:
            chunks = _split_wav(wav, CHUNK_SECONDS)
            chunks_dir = chunks[0].parent if chunks else None
        else:
            chunks = [wav]
            chunks_dir = None
        total = len(chunks)

        texts: list[str] = []
        segments: list[dict] = []
        offset = 0.0
        language = None
        align_failed = False
        any_truncated = False

        for idx, chunk in enumerate(chunks):
            # ── 取消检查点（切块边界，延迟 ≤ 一个切块）──
            if _cancelled(cancel):
                report(100.0, f"已取消（停在第 {idx}/{total} 块边界）", "".join(texts),
                       {"timing": {"kind": "cancelled", "chunk": idx, "total": total}})
                return {"text": "".join(texts), "segments": segments, "cancelled": True}

            # 防御性检查；对齐 OOM 不会卸载主模型。
            if _model is None:
                report(2.0, "重新加载模型中")
                with _lock:
                    if _model is None:
                        load_model(model_dir)
                        touch()

            base = 5.0 + idx / total * 90.0
            span = 90.0 / total

            # ── ① 抽取：切块文件已在预处理阶段就位，故恒为 0（缓存）──
            t0 = time.perf_counter()
            extract_s = time.perf_counter() - t0

            report(base, f"转录中（第 {idx + 1}/{total} 块）")

            # ── ② 识别 ──
            t1 = time.perf_counter()
            text, lang, truncated = _transcribe_one_chunk(chunk, prompt)
            recog_s = time.perf_counter() - t1

            if truncated:
                any_truncated = True
                print(f"[asr_model] ⚠ 第 {idx + 1}/{total} 块输出触顶 "
                      f"max_new_tokens={MAX_NEW_TOKENS}，疑似被截断", flush=True)
                report(base + span * 0.5, f"第 {idx + 1}/{total} 块疑似截断", "".join(texts),
                       {"level": "warn", "log": f"第 {idx + 1}/{total} 块输出触顶，疑似截断"})

            language = language or lang
            if not text:
                offset += _duration(chunk)
                continue
            texts.append(text)

            # 实时回传已识别文本（每块结束即推，不等整文件完成）
            report(base + span * 0.6, f"已识别第 {idx + 1}/{total} 块", "".join(texts))

            # ── ③ 后处理：强制对齐（仅 srt/vtt 需要）──
            t2 = time.perf_counter()
            chunk_segs = []
            chunk_duration = _duration(chunk)
            if need_timestamps and not _align_disabled:
                try:
                    chunk_segs = _align(chunk, text, lang)
                    for seg in chunk_segs:
                        seg["start"] = round(seg["start"] + offset, 3)
                        seg["end"] = round(seg["end"] + offset, 3)
                except Exception as e:
                    chunk_segs = []
                    align_failed = True
                    print(f"[asr_model] 时间戳对齐失败，使用块级近似时间戳：{e}", flush=True)
            if need_timestamps:
                if not chunk_segs:
                    align_failed = True
                    chunk_segs = [{"start": round(offset, 3),
                                   "end": round(offset + chunk_duration, 3), "text": text}]
                    report(base + span * 0.8, "对齐失败，保留块级近似字幕", "".join(texts),
                           {"level": "warn", "warning": "部分字幕使用块级近似时间戳，请检查字幕精度。",
                            "log": f"第 {idx + 1}/{total} 块没有精确时间戳，已保留整块文本和时间范围"})
                segments.extend(chunk_segs)
            post_s = time.perf_counter() - t2

            # 三段耗时日志
            report(base + span * 0.9, f"第 {idx + 1}/{total} 块完成", "".join(texts),
                   {"timing": {"kind": "chunk", "chunk": idx + 1, "total": total,
                               "extract_s": round(extract_s, 2),
                               "recog_s": round(recog_s, 2),
                               "post_s": round(post_s, 2),
                               "cached": True}})
            offset += chunk_duration

        report(96.0, "生成字幕文件", "".join(texts))
        full_text = "".join(texts).strip()
        if align_failed and not segments:
            segments = []
        if any_truncated:
            report(96.0, "完成（有块疑似截断，可调大 ASR_MAX_NEW_TOKENS）", full_text,
                   {"level": "warn", "log": "存在输出触顶的块，结果可能不完整"})
        return {"text": full_text, "segments": segments, "cancelled": False,
                "warning": "部分字幕使用块级近似时间戳，请检查字幕精度。" if align_failed else None}
    finally:
        # 只清理切块临时目录；缓存 wav 由 server 在该文件最后一个任务结束后释放
        if chunks_dir is not None:
            try:
                shutil.rmtree(chunks_dir, ignore_errors=True)
            except Exception:
                pass
        # 【显存防泄漏】每次转录结束释放缓存
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


def _align(wav: Path, transcript: str, language: str | None) -> list[dict]:
    """
    强制对齐：文本 + 音频 → 逐词时间戳 → 字幕 segments。

    v0.5.0：OOM 时**不再卸载 ASR 主模型**。
    旧实现 del _model 会导致之后每一块都从磁盘重新加载并重新 int8 量化整个 1.7B 模型，
    是远慢于正常推理的灾难性路径。现在改为：清显存重试 1 次 → 仍失败则关闭后续对齐。
    """
    global _aligner_model, _aligner_processor, _align_disabled

    if not aligner_available() or _align_disabled:
        return []

    import torch

    aligner_processor, aligner_model = load_aligner()
    if aligner_model is None:
        return []

    def _run():
        aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
            audio=str(wav), transcript=transcript, language=language,
        )
        aligner_inputs = aligner_inputs.to(aligner_model.device, torch.float16)
        with torch.inference_mode():
            outputs = aligner_model(**aligner_inputs)
        return aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=aligner_model.config.timestamp_token_id,
        )[0]

    try:
        words = _run()
    except torch.cuda.OutOfMemoryError:
        print("[asr_model] 显存不足：清理缓存后重试一次对齐（不卸载主模型）", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        try:
            words = _run()
        except torch.cuda.OutOfMemoryError as e:
            _align_disabled = True
            print(f"[asr_model] ⚠ 对齐仍显存不足，已关闭本次任务后续对齐：{e}", flush=True)
            raise

    return merge_words_to_segments(words)
