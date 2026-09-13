# -*- coding: utf-8 -*-
"""
model_registry.py — Lvats 多模型注册表（v0.6.0）

约定：
  - ./models/ 为用户模型根目录，一个子目录 = 一个模型（手动下载放进来即可）。
  - 启动 / 「重新扫描」时扫描 ./models/* 与内置 Qwen 目录，登记到 state/registry.json。
  - 目录识别规则（识别不了就跳过并打日志，不报错）：
      * 含 model.bin + tokenizer.json            → faster-whisper 引擎（CTranslate2 格式）
      * config.json architectures 含 qwen3asr 且
        不含 TokenClassification                  → qwen 引擎（对齐器目录会被自然排除）
  - registry.json 只是缓存视图，真正可信来源永远是磁盘扫描（rescan 可随时重建）。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
_TEST_MODE = os.environ.get("ASR_TEST_MODE", "").strip().lower() in ("1", "true", "yes")
REGISTRY_FILE = BASE_DIR / "state" / ("registry.test.json" if _TEST_MODE else "registry.json")

# 内置 Qwen 模型目录（现有引擎，行为不变）
QWEN_DIR_CANDIDATES = [
    "Qwen3-ASR-1.7B-hf",
    "qwen-3-asr-1.7B-Hf",
    "qwen3-asr-1.7b-hf",
]

# 1.0.0 内置 ASR 模型清单。权重不随源码发布，首次正式启动时后台下载。
BUILTIN_DOWNLOADS = [
    {"repo_id": "Qwen/Qwen3-ASR-0.6B-hf", "label": "Qwen·0.6B"},
    {"repo_id": "Qwen/Qwen3-ASR-1.7B-hf", "label": "Qwen·1.7B"},
]

# 字幕时间轴所需的配套模型，不作为 ASR 主模型选项展示。
ALIGNER_DOWNLOAD = {
    "repo_id": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
    "label": "Qwen·ForcedAligner·0.6B",
}

_REGISTRY: dict[str, dict] = {}


def _fw_label(name: str) -> str:
    n = re.sub(r"^(faster-distil-whisper-|faster-whisper-)", "", name)
    return f"FW·{n}"


def _qwen_label(name: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?[BMK]b?)", name, re.IGNORECASE)
    return f"Qwen·{m.group(1)}" if m else f"Qwen·{name}"


def _detect_fw(d: Path) -> bool:
    return (d / "model.bin").is_file() and (d / "tokenizer.json").is_file()


def _detect_qwen(d: Path) -> bool:
    cfg = d / "config.json"
    if not cfg.is_file():
        return False
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return False
    archs = " ".join(data.get("architectures") or []).lower()
    has_weights = ((d / "model.safetensors").is_file()
                   or (d / "model.safetensors.index.json").is_file()
                   or (d / "pytorch_model.bin").is_file())
    return has_weights and "qwen3asr" in archs and "tokenclassification" not in archs


def _make_entry(d: Path, engine: str) -> dict:
    return {
        "id": d.name,
        "engine": engine,
        "path": str(d.resolve()),
        "label": _fw_label(d.name) if engine == "faster-whisper" else _qwen_label(d.name),
        "ready": True,
    }


def scan() -> dict[str, dict]:
    """扫描磁盘，重建注册表。识别不了的目录跳过并打日志，不报错。"""
    reg: dict[str, dict] = {}

    def _scan_dir(d: Path, source: str):
        if _detect_fw(d):
            reg[d.name] = _make_entry(d, "faster-whisper")
        elif _detect_qwen(d):
            reg[d.name] = _make_entry(d, "qwen")
        else:
            # 只对「明显想当模型」的目录打日志（非空目录），空目录静默跳过
            try:
                if any(d.iterdir()):
                    print(f"[registry] 跳过无法识别的模型目录（{source}）：{d.name}", flush=True)
            except Exception:
                pass

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir():
            _scan_dir(d, "models/")

    for name in QWEN_DIR_CANDIDATES:
        d = BASE_DIR / name
        if d.is_dir() and _detect_qwen(d):
            reg[d.name] = _make_entry(d, "qwen")
            break

    return reg


def rescan() -> dict[str, dict]:
    """重新扫描并落盘 registry.json。"""
    global _REGISTRY
    _REGISTRY = scan()
    try:
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "saved_at": time.time(), "models": list(_REGISTRY.values()),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(REGISTRY_FILE)
    except Exception as e:
        print(f"[registry] registry.json 写入失败：{e}", flush=True)
    return _REGISTRY


def load() -> dict[str, dict]:
    """启动时调用：优先用磁盘扫描（可信来源），失败再退 registry.json。"""
    global _REGISTRY
    try:
        _REGISTRY = scan()
    except Exception as e:
        print(f"[registry] 扫描失败，尝试读缓存：{e}", flush=True)
        try:
            data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            _REGISTRY = {m["id"]: m for m in data.get("models", []) if m.get("id")}
        except Exception:
            _REGISTRY = {}
    return _REGISTRY


def get(model_id: str | None) -> dict | None:
    if not model_id:
        return None
    return _REGISTRY.get(model_id)


def default_entry() -> dict | None:
    """默认模型（v0.7.0 老板拍板）：优先 qwen 引擎中体量最小的（0.6B 优于 1.7B），
    否则取注册表第一个；都没有返回 None。"""
    def _size_b(e: dict) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)([BMK]b?)", e.get("label", ""), re.IGNORECASE)
        if not m:
            return float("inf")
        val = float(m.group(1))
        mult = {"B": 1.0, "K": 1e-6, "M": 1e-3}.get(m.group(2)[0].upper(), 1.0)
        return val * mult

    qwens = [e for e in _REGISTRY.values() if e["engine"] == "qwen"]
    if qwens:
        return sorted(qwens, key=_size_b)[0]
    for e in _REGISTRY.values():
        return e
    return None


def resolve(model_id: str | None) -> dict | None:
    """任务用的模型条目：显式指定优先，缺失/失效回退默认。"""
    return get(model_id) or default_entry()


def all_models() -> list[dict]:
    return list(_REGISTRY.values())
