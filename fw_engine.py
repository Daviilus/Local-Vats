# -*- coding: utf-8 -*-
"""
fw_engine.py — faster-whisper (CTranslate2) 引擎封装（v0.6.0）

契约（与 qwen 引擎统一，下游 srt/vtt 生成逻辑零改动）：
    transcribe(audio_path, prompt, progress_cb, cancel) ->
        {"text": str, "segments": [{"start","end","text"}], "cancelled": bool, "degraded": bool}

行为约定：
  - 懒加载：同一 model_id 复用已加载实例；切换模型前由 server 先卸载另一引擎，
    任何时刻显存内至多一个模型（qwen 或 fw 二选一）。
  - GPU 优先（cuda + int8_float16）；CUDA 不可用/加载失败 → CPU + int8，
    并通过返回值 degraded=True 明牌提示，禁止静默降级。
  - 协作式取消：whisper 按段惰性产出，切段边界检查 cancel（粒度=段，非切块）。
  - prompt（热词）为 qwen 专属能力，fw 引擎不使用（保守选择，见 DECISIONS）。
  - 进度：已完成段末时间 / 音频总时长。
"""

from __future__ import annotations

import gc
import os
import threading
import time
import asr_model

# 可调参数（环境变量覆盖，重启生效）
LANGUAGE = (os.environ.get("ASR_FW_LANGUAGE", "zh") or "zh").strip() or None
VAD_FILTER = os.environ.get("ASR_FW_VAD", "1").strip().lower() in ("1", "true", "yes")
BEAM_SIZE = int(os.environ.get("ASR_FW_BEAM_SIZE", "1"))
COMPUTE_TYPE_GPU = os.environ.get("ASR_FW_COMPUTE_TYPE_GPU", "int8_float16")
COMPUTE_TYPE_CPU = os.environ.get("ASR_FW_COMPUTE_TYPE_CPU", "int8")
# 强制设备旋钮：默认 None = 自动（cuda 优先，失败降 cpu）。
# 注意：CTranslate2 走 CUDA driver API，不理会 CUDA_VISIBLE_DEVICES，
# 需要硬性禁 GPU（如自检）时必须用本旋钮：ASR_FW_DEVICE=cpu。
DEVICE_OVERRIDE = (os.environ.get("ASR_FW_DEVICE", "") or "").strip().lower() or None

_lock = threading.RLock()
_model = None            # WhisperModel 实例
_model_id = None         # 当前加载的 registry 模型 id
_device = None           # "cuda" / "cpu"
_degraded = False        # True = 已降级 CPU（须明牌提示）
_model_path = None
_last_used_at = time.time()


def touch():
    global _last_used_at
    _last_used_at = time.time()


def seconds_since_last_use():
    return time.time() - _last_used_at


def get_device_info():
    return {"engine": "faster-whisper", "model_id": _model_id,
            "model_path": _model_path, "device": _device, "degraded": _degraded,
            "mode": "real" if is_loaded() else "unloaded",
            "compute_type": COMPUTE_TYPE_CPU if _device == "cpu" else COMPUTE_TYPE_GPU}


def is_loaded(model_id: str | None = None) -> bool:
    if _model is None:
        return False
    return model_id is None or _model_id == model_id


def current_model_id() -> str | None:
    return _model_id


def unload() -> None:
    """释放 CT2 模型对象（与 qwen 切换时保证显存内只有一个模型）。"""
    global _model, _model_id, _device, _degraded, _model_path
    with asr_model.INFERENCE_LOCK, _lock:
        if _model is None:
            return
        _model = None
        _model_id = None
        _model_path = None
        _device = None
        _degraded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _load(model_path: str, model_id: str, progress_cb=None):
    with asr_model.INFERENCE_LOCK, _lock:
        if asr_model.USE_PLACEHOLDER:
            return
        if _model is not None and _model_id == model_id and _model_path == model_path:
            touch()
            return
        unload()
        _load_new(model_path, model_id, progress_cb)
        touch()


def preload(model_entry, progress_cb=None):
    _load(str(model_entry['path']), model_entry['id'], progress_cb)


def _load_new(model_path: str, model_id: str, progress_cb=None):
    """加载 WhisperModel：GPU 优先，失败降级 CPU（明牌）。"""
    global _model, _model_id, _device, _degraded, _model_path
    from faster_whisper import WhisperModel

    def report(msg, warn=False):
        if progress_cb:
            try:
                progress_cb(1.0, msg, None,
                            {"level": "warn" if warn else "t", "log": msg,
                             "fw_degraded": warn})
            except Exception:
                pass

    if progress_cb:
        progress_cb(1.0, f"加载模型中（{model_id}）", None, {"log": f"[fw] 加载 {model_id}"})
    with _lock:
        if _model is not None and _model_id == model_id:
            return
        # 双保险：进入加载前确另一个引擎不在显存（server 侧也会先卸载）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        last_err = None
        if DEVICE_OVERRIDE:
            plan = [(DEVICE_OVERRIDE,
                     COMPUTE_TYPE_CPU if DEVICE_OVERRIDE == "cpu" else COMPUTE_TYPE_GPU,
                     DEVICE_OVERRIDE == "cpu")]
        else:
            plan = [("cuda", COMPUTE_TYPE_GPU, False), ("cpu", COMPUTE_TYPE_CPU, True)]
        for device, ctype, degraded in plan:
            try:
                m = WhisperModel(model_path, device=device, compute_type=ctype)
                _model, _model_id, _device, _degraded = m, model_id, device, degraded
                _model_path = model_path
                report(f"模型已加载（faster-whisper / {device} / {ctype}）"
                       + ("　⚠ 已降级 CPU：GPU 不可用或显存不足" if degraded else ""))
                if degraded:
                    report("⚠ faster-whisper 已降级 CPU + int8（GPU 不可用或显存不足）", warn=True)
                return
            except Exception as e:
                last_err = e
                print(f"[fw_engine] {device} 加载失败：{e}", flush=True)
        raise RuntimeError(f"faster-whisper 模型加载失败（cuda/cpu 均尝试）：{last_err}")


def transcribe(audio_path, model_entry: dict, progress_cb=None, cancel=None) -> dict:
    if asr_model.USE_PLACEHOLDER:
        return asr_model._transcribe_placeholder(None, progress_cb, cancel, True)
    with asr_model.INFERENCE_LOCK:
        try:
            touch()
            return _transcribe_locked(audio_path, model_entry, progress_cb, cancel)
        finally:
            touch()


def _transcribe_locked(audio_path, model_entry: dict, progress_cb=None, cancel=None) -> dict:
    """
    主入口。audio_path: Path；model_entry: registry 条目。
    返回统一契约；cancelled=True 时 text/segments 为已产出部分（不生成文件，与 qwen 一致）。
    """
    if _model is None or _model_id != model_entry["id"] or _model_path != str(model_entry['path']):
        if _cancelled(cancel):
            return {"text": "", "segments": [], "cancelled": True, "degraded": False}
        _load(str(model_entry["path"]), model_entry["id"], progress_cb)
    if _cancelled(cancel):
        return {"text": "", "segments": [], "cancelled": True, "degraded": False}

    if _degraded:
        # 降级信息必须带到任务行（不静默）
        if progress_cb:
            try:
                progress_cb(2.0, "⚠ faster-whisper 已降级 CPU + int8", None,
                            {"level": "warn", "fw_degraded": True,
                             "log": "⚠ fw 引擎 CPU 降级：GPU 不可用或显存不足"})
            except Exception:
                pass

    try:
        from asr_model import _duration  # 复用 ffprobe 时长探测（不重复造轮子）
        total = _duration(audio_path) or 0.0
    except Exception:
        total = 0.0

    t0 = time.perf_counter()
    try:
        segments_iter, info = _model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            vad_filter=VAD_FILTER,
            beam_size=BEAM_SIZE,
        )
    except Exception as e:
        raise RuntimeError(f"faster-whisper 推理失败：{e}")

    texts: list[str] = []
    segments: list[dict] = []
    cancelled = False
    try:
        for seg in segments_iter:
            if _cancelled(cancel):
                cancelled = True
                if progress_cb:
                    progress_cb(100.0, "已取消（段边界停止）", "".join(texts),
                                {"timing": {"kind": "cancelled",
                                            "chunk": len(segments),
                                            "total": max(len(segments), 1)}})
                break
            item = {"start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": (seg.text or "").strip()}
            if not item["text"]:
                continue
            segments.append(item)
            texts.append(item["text"])
            if progress_cb:
                pct = min(99.0, (seg.end / total * 95.0) if total > 0 else
                          min(95.0, len(segments) * 5.0))
                progress_cb(pct, f"转录中（已产出 {len(segments)} 段，{item['end']:.0f}s/"
                                 f"{total:.0f}s）", "".join(texts))
    except Exception as e:
        raise RuntimeError(f"faster-whisper 段迭代失败：{e}")

    full_text = "".join(texts).strip()
    if progress_cb and not cancelled:
        progress_cb(100.0, "完成", full_text,
                    {"log": f"[fw] 完成：{len(segments)} 段，耗时 {time.perf_counter()-t0:.1f}s"})
    return {"text": full_text, "segments": segments, "cancelled": cancelled,
            "degraded": _degraded}


def _cancelled(cancel) -> bool:
    try:
        return bool(cancel is not None and cancel.is_set())
    except Exception:
        return False
