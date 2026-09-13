# -*- coding: utf-8 -*-
"""
server.py — Lvats · Local Video/Audio Transform Service v0.5.3

v1.0.0 变更：首个 GitHub 源码发行版，补齐公共安装说明与通用虚拟环境启动支持。
v0.9.5 变更：文件和文件夹原生选择框直接设置 HWND_TOPMOST，修复后台弹窗未置顶。
v0.9.4 变更：提示词输入区域的垂直高度扩大为原来的两倍。
v0.9.3 变更：已归档任务模块新增可记忆状态的折叠按钮。
v0.9.2 变更：新增可靠的本机服务重启，重启前原子保存队列，由独立辅助进程复用 Windows 启动链路拉起服务。
v0.9.1 变更：文件队列/任务队列可折叠；完成任务可持久化归档并在独立模块查看、恢复。
v0.9.0 变更：修复 Windows PowerShell 5.1 读取启动脚本的 UTF-8 编码，以及 BAT 传递项目根目录参数的尾部反斜杠解析。
v0.8.6 变更：HTTP/HTTPS 同端口分流重定向；修复重复启动竞争导致连到旧的非 GPU 实例。
v0.8.5 变更：证书初始化兼容 Windows PowerShell 5.1，启动器不再依赖 pwsh。
v0.8.4 变更：正式服务默认启用受本机信任的 HTTPS；测试模式继续使用 HTTP。
v0.8.3 变更：文件队列每页 10 个，后端统一限制最多 50 个文件。

v0.5.3 变更：
  A 服务正式命名 Lvats：界面/控制台窗口标题/pid 文件（lvats.pid）/启动日志统一可辨认
  B 新增 POST /api/shutdown + 页面「停止服务」按钮（二次确认，响应先回前端，几秒内退出）
  C 任务行内进度条补回（百分比+耗时走原有轮询/SSE 通道）

v0.5.0 变更（保留）：
  A 速度：wav 按【文件】缓存复用；显式贪心解码 + max_new_tokens 上限；
          device_map 显式 cuda（禁止静默 CPU 降级）；每块三段耗时日志；
          txt 任务跳过 ForcedAligner；修复对齐 OOM 导致每块重载模型的致命隐患
  B 显示：EventBus + /api/events（SSE）推送日志、每块实时文本、三段耗时、设备实况
  C 功能：任务删除 / 协作式取消 / ↑↓ 挪动 / 队列持久化 json
  D 对话框置顶：隐藏 TopMost 属主窗体 + ShowDialog(属主)，关闭后立即销毁属主

端口：默认 8000，可用 ASR_PORT 覆盖（自检用 8001）。
自检：ASR_PLACEHOLDER=1 时全程不加载真实模型，不占显存。
测试隔离：ASR_TEST_MODE=1 时 output / queue 文件自动加 .test 后缀，不污染正式数据。

监听 127.0.0.1:{ASR_PORT}（正式端口默认 8000，固定）。
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr_model  # 真实模型推理（int8/fp16 量化 + 强制对齐时间戳）
import fw_engine  # v0.6.0：faster-whisper 引擎（faster_whisper 缺失时延迟报错，不影响启动）
import model_registry  # v0.6.0：多模型注册表（./models/ 扫描 + registry.json）

# stdout/stderr 重定向到文件/管道时默认跟随系统 ANSI 编码（cp936），
# 打印 ▶ ✓ ■ ⚠ 等字符会 UnicodeEncodeError 并炸掉 worker 协程 —— 统一强制 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Windows UCRT 不认识 IANA 时区名：TZ=Asia/Shanghai 这类值会被错误解析
# （实测 localtime 偏移 -7h），污染产物文件名时间戳与日志时间。
# 系统时区本身正确，必须在进程首次 localtime 调用前清掉 TZ（含 "/" 即 IANA 名，POSIX 格式不含）。
if os.name == "nt" and "/" in os.environ.get("TZ", ""):
    os.environ.pop("TZ", None)

VERSION = "1.0.0"
SERVICE_NAME = "Lvats"
SERVICE_FULLNAME = "Lvats · Local Video/Audio Transform Service"

BASE_DIR = Path(__file__).resolve().parent

# 测试隔离：ASR_TEST_MODE=1 → output / queue 加 .test 后缀
TEST_MODE = os.environ.get("ASR_TEST_MODE", "").strip().lower() in ("1", "true", "yes")
_SUF = ".test" if TEST_MODE else ""

OUTPUT_DIR = BASE_DIR / f"output{_SUF}"
UPLOAD_DIR = BASE_DIR / f".dropcache{_SUF}"
STATIC_DIR = BASE_DIR / "static"
CONFIG_DIR = BASE_DIR / "config"
STATE_DIR = BASE_DIR / "state"
QUEUE_FILE = STATE_DIR / f"queue{_SUF}.json"
PID_FILE = STATE_DIR / f"lvats{_SUF}.pid"
RESTART_STATUS_FILE = STATE_DIR / f"restart{_SUF}.json"
RESTART_HELPER = BASE_DIR / "scripts" / "restart_lvats.py"
QUICK_PROMPTS_FILE = CONFIG_DIR / f"quick_prompts{_SUF}.json"
MAX_QUICK_PROMPTS = 20
MAX_FILES = 50
PORT = int(os.environ.get("ASR_PORT", "8000"))
HTTPS_ENABLED = os.environ.get("ASR_HTTPS", "0" if TEST_MODE else "1").strip().lower() in ("1", "true", "yes")
HTTP_REDIRECT_ENABLED = os.environ.get("ASR_HTTP_REDIRECT", "1" if HTTPS_ENABLED else "0").strip().lower() in ("1", "true", "yes")
TLS_BACKEND_PORT = int(os.environ.get("ASR_TLS_BACKEND_PORT", str(PORT + 443)))
SSL_DIR = Path(os.environ.get("ASR_SSL_DIR") or (CONFIG_DIR / "ssl"))
SSL_CERTFILE = Path(os.environ.get("ASR_SSL_CERTFILE") or (SSL_DIR / "lvats-cert.pem"))
SSL_KEYFILE = Path(os.environ.get("ASR_SSL_KEYFILE") or (SSL_DIR / "lvats-key.pem"))
SERVICE_SCHEME = "https" if HTTPS_ENABLED else "http"
SERVICE_URL = f"{SERVICE_SCHEME}://127.0.0.1:{PORT}"

# 持久化字段白名单（_token 等运行时对象不落盘）
PERSIST_FIELDS = ("id", "path", "name", "format", "prompt", "status", "progress",
                  "partial_text", "warning", "error", "output_path", "output_name",
                  "created_at", "started_at", "finished_at", "duration", "message",
                  "cancel_requested", "model_id", "managed_upload", "archived", "archived_at")

# ──────────────────────────────────────────────────────────────
# 状态表
# ──────────────────────────────────────────────────────────────
FILES: dict[str, dict] = {}   # file_id -> {id, path, name, ok, reason, managed_upload}
TASKS: dict[str, dict] = {}   # task_id -> {...}
TASK_QUEUE: "TaskQueue | None" = None
BUS: "EventBus | None" = None
_dirty = False
_preload_task = None
_files_lock = threading.Lock()
_drop_reservations = 0


def _set_console_title(title: str):
    """Windows：设置控制台窗口标题，任务栏一眼可认（非 Windows / 无控制台时静默跳过）。"""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    except Exception:
        pass


def _try_set_proc_name(name: str):
    """环境有 setproctitle 则修改进程名；没有不强求。"""
    try:
        import setproctitle  # type: ignore
        setproctitle.setproctitle(name)
    except Exception:
        pass


def _uvicorn_kwargs(port: int | None = None) -> dict:
    """构造正式/测试服务启动参数；正式 HTTPS 缺证书时拒绝静默降级为 HTTP。"""
    options = {"host": "127.0.0.1", "port": PORT if port is None else port}
    if not HTTPS_ENABLED:
        return options
    missing = [path for path in (SSL_CERTFILE, SSL_KEYFILE) if not path.is_file()]
    if missing:
        names = "、".join(str(path) for path in missing)
        raise RuntimeError(f"HTTPS 证书未就绪：{names}；请先运行 scripts/setup_local_https.ps1")
    options.update(ssl_certfile=str(SSL_CERTFILE), ssl_keyfile=str(SSL_KEYFILE))
    return options


def _write_pid_file():
    """写 lvats.pid（pid / 端口 / 启动时间），供脚本与用户辨认进程；优雅停止时删除。"""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(json.dumps({
            "name": SERVICE_NAME, "pid": os.getpid(), "port": PORT,
            "version": VERSION,
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[{SERVICE_NAME}] pid 文件写入失败：{e}", flush=True)


def _remove_owned_pid_file() -> bool:
    """仅删除当前进程写入的 PID 文件，避免端口竞争失败者删掉正在服务的实例记录。"""
    try:
        if not PID_FILE.is_file():
            return False
        payload = json.loads(PID_FILE.read_text(encoding="utf-8"))
        if payload.get("pid") != os.getpid():
            return False
        PID_FILE.unlink()
        return True
    except Exception:
        return False


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p).strip().strip('"')))


def _check_path(p: str) -> tuple[bool, str]:
    path = Path(p)
    if not path.exists():
        return False, "路径不存在"
    if not path.is_file():
        return False, "不是文件"
    if not os.access(path, os.R_OK):
        return False, "不可读"
    return True, ""


def _is_inside_upload_dir(path: str | Path) -> bool:
    """只允许清理 Lvats 自己的拖放缓存，绝不触碰用户原文件。"""
    try:
        Path(path).resolve().relative_to(UPLOAD_DIR.resolve())
        return True
    except (OSError, ValueError):
        return False


def _remove_managed_upload(path: str | Path) -> bool:
    if not _is_inside_upload_dir(path):
        return False
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError as exc:
        _log("warn", f"拖放临时副本清理失败：{exc}")
        return False


def _cleanup_managed_if_unused(path: str | Path, managed: bool) -> bool:
    """文件行和活动任务都不再引用时，安全删除拖放副本。"""
    if not managed:
        return False
    normalized = _norm(path)
    if any(_norm(f.get("path", "")) == normalized for f in FILES.values()):
        return False
    if any(_norm(t.get("path", "")) == normalized and t.get("status") in ("queued", "running")
           for t in TASKS.values()):
        return False
    return _remove_managed_upload(path)


def _cleanup_orphan_uploads():
    """启动时保留待执行任务的副本，清除中断上传和无引用残留。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    retained = {
        _norm(t.get("path", "")) for t in TASKS.values()
        if t.get("managed_upload") and t.get("status") in ("queued", "running")
    }
    removed = 0
    for candidate in UPLOAD_DIR.iterdir():
        if candidate.is_file() and _norm(candidate) not in retained:
            try:
                candidate.unlink()
                removed += 1
            except OSError as exc:
                _log("warn", f"拖放缓存清理失败：{candidate.name} — {exc}")
    if removed:
        _log("t", f"已清理 {removed} 个无引用拖放临时文件")


def _elapsed_text(seconds: float) -> str:
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _task_elapsed(t: dict) -> tuple[str, float]:
    """返回 (显示文案, 秒数)。已结束任务用 finished_at - started_at 固化，不再累加。"""
    st = t.get("started_at")
    if st is None:
        return "0:00:00", 0.0
    end = t.get("finished_at") or (None if t["status"] == "running" else None)
    secs = (end - st) if end else (time.time() - st)
    secs = max(0.0, secs)
    return _elapsed_text(secs), secs


# ──────────────────────────────────────────────────────────────
# 事件总线（SSE）
# ──────────────────────────────────────────────────────────────
class EventBus:
    """
    线程安全的事件总线。
    publish() 可能从 asyncio.to_thread 的工作线程调用，
    因此用 loop.call_soon_threadsafe 跨线程桥回 event loop。
    """

    def __init__(self, history: int = 500):
        self._subs: set[asyncio.Queue] = set()
        self._hist: deque = deque(maxlen=history)
        self._seq = 0
        self._publish_lock = threading.Lock()
        self.stream_id = uuid.uuid4().hex
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop):
        self._loop = loop

    def publish(self, kind: str, **data) -> dict:
        with self._publish_lock:
            self._seq += 1
            ev = {"seq": self._seq, "ts": time.time(), "kind": kind, **data}
            self._hist.append(ev)
            loop = self._loop
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(self._fanout, ev)
                except RuntimeError:
                    pass
            return ev

    def _fanout(self, ev: dict):
        for q in list(self._subs):
            try:
                if q.qsize() < 200:
                    q.put_nowait(ev)
            except Exception:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subs.add(q)
        return q

    def unsubscribe(self, q):
        self._subs.discard(q)

    def since(self, seq: int) -> list[dict]:
        with self._publish_lock:
            return [e for e in self._hist if e["seq"] > seq]


def _log(level: str, msg: str, **extra):
    """写终端 + 推 SSE 日志事件。level: t / ok / err / warn"""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except Exception:
        pass  # 日志失败绝不能拖死 worker（防御 GBK/管道等异常环境）
    if BUS:
        BUS.publish("log", level=level, text=msg, **extra)


def _fmt_timing(tm: dict) -> str:
    kind = tm.get("kind")
    if kind == "prep":
        return f"[prep] 抽取16k wav {tm.get('prep_s', 0):.1f}s（{'缓存命中' if tm.get('cache_hit') else '新建'}）"
    if kind == "chunk":
        return (f"[chunk {tm.get('chunk')}/{tm.get('total')}] "
                f"抽取 {tm.get('extract_s', 0):.1f}s(缓存) | "
                f"识别 {tm.get('recog_s', 0):.1f}s | "
                f"后处理 {tm.get('post_s', 0):.1f}s")
    if kind == "cancelled":
        return f"[cancel] 已在第 {tm.get('chunk')}/{tm.get('total')} 块边界停止"
    return ""


# ──────────────────────────────────────────────────────────────
# 任务队列（显式有序列表，支持删除 / 挪动 / 队首消费）
# ──────────────────────────────────────────────────────────────
class TaskQueue:
    """
    只存「排队中」的 task_id。running 任务在 pop 时就已出队，
    因此「不能移到运行中任务之前」天然成立，无需额外约束。
    所有操作都在 event loop 单线程内执行（端点均为 async def），无需加锁。
    """

    def __init__(self):
        self._order: list[str] = []
        self._wake = asyncio.Event()

    def put(self, tid: str):
        self._order.append(tid)
        self._wake.set()

    def remove(self, tid: str) -> bool:
        if tid in self._order:
            self._order.remove(tid)
            return True
        return False

    def move(self, tid: str, delta: int) -> bool:
        try:
            i = self._order.index(tid)
        except ValueError:
            return False
        j = i + delta
        if not (0 <= j < len(self._order)):
            return False
        self._order[i], self._order[j] = self._order[j], self._order[i]
        return True

    async def pop_next(self) -> str:
        """永远取队首；队列空则挂起等待。"""
        while True:
            if not self._order:
                self._wake.clear()
                await self._wake.wait()
                continue
            tid = self._order.pop(0)
            t = TASKS.get(tid)
            if t is not None and t["status"] == "queued":
                return tid
            # 已被删除 / 取消的任务直接跳过

    def snapshot(self) -> list[str]:
        return list(self._order)

    def load(self, ids: list[str]):
        self._order = [i for i in ids if i in TASKS]
        if self._order:
            self._wake.set()


class CancelToken:
    """协作式取消令牌：跨线程传给阻塞在 to_thread 里的同步推理函数。"""

    def __init__(self):
        self._e = threading.Event()

    def set(self):
        self._e.set()

    def is_set(self) -> bool:
        return self._e.is_set()


# ──────────────────────────────────────────────────────────────
# 字幕渲染
# ──────────────────────────────────────────────────────────────
def _fmt_time(seconds: float, sep: str) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render_txt(result: dict) -> str:
    return result.get("text", "").strip() + "\n"


def render_srt(result: dict) -> str:
    lines = []
    for idx, seg in enumerate(result["segments"], start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt_time(seg['start'], ',')} --> {_fmt_time(seg['end'], ',')}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def render_vtt(result: dict) -> str:
    lines = ["WEBVTT", ""]
    for seg in result["segments"]:
        lines.append(f"{_fmt_time(seg['start'], '.')} --> {_fmt_time(seg['end'], '.')}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def sanitize_filename(value: str) -> str:
    value = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", value)
    value = value.strip().strip(".")
    return value or "file"


def make_unique_path(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 1000):
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("同名文件过多")


def save_result(original_filename: str, output_format: str, result: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(Path(original_filename).stem)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{stem}_转录_{stamp}.{output_format}"
    out_path = make_unique_path(OUTPUT_DIR, filename)
    renderers = {"txt": render_txt, "srt": render_srt, "vtt": render_vtt}
    out_path.write_text(renderers[output_format](result), encoding="utf-8")
    return out_path


# ──────────────────────────────────────────────────────────────
# 队列持久化
# ──────────────────────────────────────────────────────────────
def _serializable(t: dict) -> dict:
    d = {k: t.get(k) for k in PERSIST_FIELDS}
    if isinstance(d.get("partial_text"), str):
        d["partial_text"] = d["partial_text"][:20000]
    return d


def mark_dirty():
    global _dirty
    _dirty = True


def persist_now(strict: bool = False) -> bool:
    global _dirty
    try:
        payload = {"version": VERSION, "saved_at": time.time(),
                   "order": TASK_QUEUE.snapshot() if TASK_QUEUE else [],
                   "tasks": [_serializable(t) for t in TASKS.values()]}
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = QUEUE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, QUEUE_FILE)
        _dirty = False
        return True
    except Exception as e:
        print(f"[server] 队列持久化失败：{e}", flush=True)
        if strict:
            raise RuntimeError(f"队列持久化失败：{e}") from e
        return False


async def _persist_loop():
    global _dirty
    while True:
        await asyncio.sleep(1.5)
        if _dirty:
            persist_now()


def _has_other_active(path: str, exclude: str) -> bool:
    p = _norm(path)
    return any(t["path"] == p and t["id"] != exclude
               and t["status"] in ("queued", "running") for t in TASKS.values())


def restore_queue():
    """启动时恢复队列。running → queued（保守：重跑），且排到队首（它原本正在跑）。"""
    if not QUEUE_FILE.exists():
        return 0
    try:
        data = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[server] 队列文件损坏，已忽略：{e}", flush=True)
        return 0

    reset_running: list[str] = []
    for rec in data.get("tasks", []):
        tid = rec.get("id")
        if not tid:
            continue
        rec.setdefault("created_at", time.time())
        rec.setdefault("progress", 0.0)
        rec.setdefault("partial_text", "")
        rec.setdefault("archived", False)
        rec.setdefault("archived_at", None)
        if rec.get("status") == "running":
            rec["status"] = "queued"
            rec["started_at"] = None
            rec["finished_at"] = None
            rec["progress"] = 0.0
            rec["partial_text"] = ""
            rec["message"] = "重启前为「运行中」，已重置为排队中（将重新转录）"
            reset_running.append(tid)
        if rec.get("status") == "queued":
            ok, reason = _check_path(rec.get("path", ""))
            if not ok:
                rec["status"] = "failed"
                rec["error"] = f"重启后源文件不可访问：{reason}"
        TASKS[tid] = rec

    # 优先尊重落盘的队列顺序（用户可能挪动过）；没有保存顺序才按创建时间兜底
    saved_order = [tid for tid in data.get("order", []) if tid in TASKS]
    order = [tid for tid in saved_order if TASKS[tid]["status"] == "queued"]
    for tid, t in TASKS.items():
        if t["status"] == "queued" and tid not in order:
            order.append(tid)
    if not saved_order:
        order.sort(key=lambda tid: TASKS[tid].get("created_at") or 0)
    else:
        # 重启前正在跑的任务排到队首（保持彼此相对顺序），符合「接着上次跑」的直觉
        for tid in reversed(reset_running):
            if tid in order:
                order.remove(tid)
                order.insert(0, tid)
    if TASK_QUEUE:
        TASK_QUEUE.load(order)
    return len(TASKS)


# ──────────────────────────────────────────────────────────────
# worker：单消费者顺序执行；支持协作式取消
# ──────────────────────────────────────────────────────────────
def _make_cb(tid: str):
    def cb(pct, msg, partial=None, meta=None):
        t = TASKS.get(tid)
        if t is None:
            return
        t["progress"] = round(float(pct), 1)
        t["message"] = msg
        if partial is not None:
            t["partial_text"] = partial
            BUS.publish("partial", task_id=tid, text=partial)
        meta = meta or {}
        if meta.get("warning"):
            t["warning"] = meta["warning"]
        if meta.get("fw_degraded"):
            t["warning"] = "⚠ faster-whisper 已降级 CPU + int8（GPU 不可用或显存不足）"
        if meta.get("device"):
            BUS.publish("device", task_id=tid, **meta["device"])
            if meta["device"].get("degraded"):
                t["warning"] = f"⚠ CPU 降级：部分层被放到 CPU（{meta['device'].get('cpu_layers')}）"
        if meta.get("log"):
            _log(meta.get("level", "t"), f"{meta['log']}  [{t['name']}]", task_id=tid)
        tm = meta.get("timing")
        if tm:
            # 结构化耗时单独发一路（注意：tm 自带 kind 字段，必须包在子字典里，
            # 否则会把 SSE 的事件名 "timing" 覆盖成 "chunk"/"prep"）
            BUS.publish("timing", task_id=tid, name=t["name"], timing=tm)
            line = _fmt_timing(tm)
            if line:
                _log("t", f"{line}  [{t['name']}]", task_id=tid)
        mark_dirty()
    return cb


def _finish(t: dict, status: str, error: str | None = None, partial: str | None = None):
    t["status"] = status
    t["error"] = error
    if partial is not None:
        t["partial_text"] = partial
    st = t.get("started_at")
    t["finished_at"] = time.time()
    t["duration"] = round((t["finished_at"] - st), 1) if st else 0.0
    if status != "running":
        t["progress"] = 100.0 if status == "done" else t.get("progress", 0.0)
    mark_dirty()


async def _run_engine(t: dict, tid: str, tok, cb) -> dict:
    """
    v0.6.0 多引擎分发：按任务 model_id 从注册表解析引擎。
    任何时刻显存内至多一个模型：切引擎前先卸载另一个（qwen↔fw 互斥）。
    异常直接抛出，由 worker 统一置 failed。
    """
    entry = model_registry.resolve(t.get("model_id"))
    if entry is None:
        raise RuntimeError("注册表中没有任何可用模型（请检查 ./models/ 与 Qwen 目录）")
    if entry["id"] != t.get("model_id"):
        _log("warn", f"任务指定模型 {t.get('model_id')} 不可用，回退默认 {entry['label']}",
             task_id=tid)
        t['warning'] = f"指定模型不可用，已回退 {entry['label']}"
        t['model_id'] = entry['id']
        mark_dirty()
    return await asyncio.to_thread(_run_engine_sync, entry, t, tid, tok, cb)


def _run_engine_sync(entry, t, tid, tok, cb):
    # 切换引擎与整个推理过程在同一个锁内，预加载不能插入并占用显存。
    with asr_model.INFERENCE_LOCK:
        return _transcribe_selected(entry, t, tid, tok, cb)


def _transcribe_selected(entry, t, tid, tok, cb):
    if entry["engine"] == "faster-whisper":
        if asr_model.is_loaded():
            asr_model.unload_model()
            _log("t", f"[多模型] 已卸载 Qwen 模型，切换到 {entry['label']}", task_id=tid)
        result = fw_engine.transcribe(Path(t["path"]), entry, cb, tok)
        if result.get("degraded"):
            t["warning"] = "⚠ faster-whisper 已降级 CPU + int8（GPU 不可用或显存不足）"
        return result
    # qwen 引擎（行为不变，回归基准）
    if fw_engine.is_loaded():
        fw_engine.unload()
        _log("t", f"[多模型] 已卸载 faster-whisper 模型，切换到 {entry['label']}", task_id=tid)
    return asr_model.transcribe_audio(
        Path(t["path"]), t.get("prompt") or None,
        cb, tok, t["format"] in ("srt", "vtt"), tid,
        model_dir=entry['path'],
    )


async def _worker():
    while True:
        tid = await TASK_QUEUE.pop_next()
        t = TASKS.get(tid)
        if t is None:
            continue

        tok = CancelToken()
        t["_token"] = tok
        t["status"] = "running"
        t["started_at"] = time.time()
        t["finished_at"] = None
        t["progress"] = 0.0
        t["error"] = None
        t["cancel_requested"] = False
        mark_dirty()

        BUS.publish("task_start", task_id=tid, name=t["name"], format=t["format"])
        _log("ok", f"▶ 开始：{t['name']} [{t['format']}]", task_id=tid)

        ok, reason = _check_path(t["path"])
        if not ok:
            _finish(t, "failed", error=f"文件复查失败：{reason}")
            _log("err", f"✗ 失败：{t['name']} — {reason}", task_id=tid)
        else:
            try:
                result = await _run_engine(t, tid, tok, _make_cb(tid))
                if result.get("cancelled"):
                    _finish(t, "cancelled", partial=result.get("text", ""))
                    _log("warn", f"■ 已取消：{t['name']}（保留已识别部分，未生成文件）", task_id=tid)
                else:
                    warning = result.get('warning')
                    segments = result.get("segments") or []
                    if not segments and result.get('text', '').strip() and t["format"] in ("srt", "vtt"):
                        duration = await asyncio.to_thread(asr_model._duration, Path(t['path']))
                        if duration <= 0:
                            raise RuntimeError("模型未返回时间戳，且无法读取音频时长，不能生成有效字幕。")
                        warning = "模型未返回精确时间戳，已按整段音频时长保留文本，请检查字幕精度。"
                        result["segments"] = [{"start": 0.0, "end": duration,
                                               "text": result.get("text", "")}]
                    if warning:
                        _log('warn', warning, task_id=tid)
                    out = save_result(t["name"], t["format"], result)
                    t["output_path"] = str(out)
                    t["output_name"] = out.name
                    t["full_text"] = result.get("text", "")
                    t["partial_text"] = result.get("text", "")
                    t["warning"] = warning or t.get("warning")   # 淇濈暀寮撴搸宸茶缃疆殑闄嶇骇鍛婅
                    _finish(t, "done")
                    _log("ok", f"✓ 完成：{t['name']} → {out.name}", task_id=tid,
                         output_path=str(out))
            except Exception as e:
                _finish(t, "failed", error=str(e))
                _log("err", f"✗ 失败：{t['name']} — {e}", task_id=tid)

        BUS.publish("task_end", task_id=tid, status=t["status"])
        t.pop("_token", None)

        # 该文件最后一个任务结束 → 删除 wav 缓存；还有别的任务在用则只解除引用
        try:
            asr_model.wav_cache.release(Path(t["path"]), tid,
                                        keep=_has_other_active(t["path"], tid))
        except Exception:
            pass
        _cleanup_managed_if_unused(t["path"], bool(t.get("managed_upload")))
        persist_now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global TASK_QUEUE, BUS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    BUS = EventBus()
    BUS.bind_loop(asyncio.get_running_loop())
    TASK_QUEUE = TaskQueue()

    model_registry.load()
    _log("t", f"[多模型] 注册表就绪：{len(model_registry.all_models())} 个模型（"
               f"{', '.join(m['label'] for m in model_registry.all_models()) or '无'}）")
    if not TEST_MODE and not asr_model.USE_PLACEHOLDER:
        _queue_initial_model_downloads()

    # v0.6.0a：CUDA 环境启动自检（防止误用 CPU 版 torch 的 Python 启动而毫无提示）
    try:
        import torch
        _cuda_ok = torch.cuda.is_available()
        _torch_ver = torch.__version__
    except Exception:
        _cuda_ok, _torch_ver = False, "未安装"
    if not _cuda_ok:
        _log("err", "⚠ CUDA 不可用：当前 Python 的 torch 可能是 CPU 版（" + _torch_ver +
                   "）或 GPU 被占用/驱动异常。请用 启动Lvats.bat 启动（它锁定带 CUDA torch 的运行时）。")
    else:
        _log("t", f"[CUDA] 可用（torch {_torch_ver}）")

    _set_console_title(SERVICE_NAME)
    _try_set_proc_name(SERVICE_NAME)
    _write_pid_file()

    asr_model.wav_cache.drop_all()          # 启动即清空 wav 缓存，绝不留脏数据

    n = restore_queue()
    _cleanup_orphan_uploads()
    if n:
        _log("ok", f"已从 {QUEUE_FILE.name} 恢复 {n} 条任务记录")

    background = [asyncio.create_task(_worker(), name="asr-worker"),
                  asyncio.create_task(_persist_loop(), name="asr-persist"),
                  asyncio.create_task(_idle_models_loop(), name="models-idle")]
    _log("ok", f"[{SERVICE_NAME}] v{VERSION} 就绪：{SERVICE_URL}"
               + ("（占位模式，不加载模型）" if asr_model.USE_PLACEHOLDER else "")
               + ("（测试隔离模式）" if TEST_MODE else ""))
    yield
    for task in background:
        task.cancel()
    await asyncio.gather(*background, return_exceptions=True)
    persist_now()
    _cleanup_orphan_uploads()
    _remove_owned_pid_file()


app = FastAPI(title=SERVICE_FULLNAME, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ──────────────────────────────────────────────────────────────
# 原生文件 / 文件夹选择对话框（v0.5.0：隐藏 TopMost 属主窗体承载模态对话框）
# ──────────────────────────────────────────────────────────────
ALLOWED_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".ts", ".m4v",
}

# 属主窗体保持隐藏；实际 Explorer 公共对话框出现后，再按进程枚举其 HWND 并直接置顶。
# 仅设置 owner.TopMost 并不可靠：后台 PowerShell 创建的公共对话框可能不继承 WS_EX_TOPMOST。
PS_OWNER_PREAMBLE = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

public static class LvatsDialogTopmost {
    private delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);
    private static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
    private const uint SWP_NOSIZE = 0x0001;
    private const uint SWP_NOMOVE = 0x0002;
    private static readonly HashSet<IntPtr> Activated = new HashSet<IntPtr>();
    private static Thread PinThread;
    private static volatile bool Pinning;

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);
    [DllImport("kernel32.dll")]
    private static extern uint GetCurrentProcessId();
    [DllImport("user32.dll")]
    private static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SetWindowPos(
        IntPtr hwnd, IntPtr insertAfter, int x, int y, int width, int height, uint flags);
    [DllImport("user32.dll")]
    private static extern bool BringWindowToTop(IntPtr hwnd);
    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hwnd);

    public static uint CurrentProcessId() {
        return GetCurrentProcessId();
    }

    public static void PinDialogs(uint processId, IntPtr owner) {
        EnumWindows(delegate(IntPtr hwnd, IntPtr lParam) {
            uint ownerProcessId;
            GetWindowThreadProcessId(hwnd, out ownerProcessId);
            if (ownerProcessId != processId || hwnd == owner || !IsWindowVisible(hwnd)) return true;
            bool first;
            lock (Activated) {
                first = Activated.Add(hwnd);
            }
            SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
            if (first) {
                BringWindowToTop(hwnd);
                SetForegroundWindow(hwnd);
            }
            return true;
        }, IntPtr.Zero);
    }

    public static void Start(uint processId, IntPtr owner) {
        Stop();
        lock (Activated) {
            Activated.Clear();
        }
        Pinning = true;
        PinThread = new Thread(delegate() {
            while (Pinning) {
                try {
                    PinDialogs(processId, owner);
                } catch {
                    // The dedicated picker process exits immediately after the dialog closes.
                }
                Thread.Sleep(50);
            }
        });
        PinThread.IsBackground = true;
        PinThread.Start();
    }

    public static void Stop() {
        Pinning = false;
        Thread thread = PinThread;
        PinThread = null;
        if (thread != null && thread != Thread.CurrentThread) thread.Join(500);
    }
}

public static class LvatsFolderPicker {
    private delegate int BrowseCallbackProc(
        IntPtr hwnd, uint message, IntPtr lParam, IntPtr data);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    private struct BrowseInfo {
        public IntPtr owner;
        public IntPtr root;
        public IntPtr displayName;
        [MarshalAs(UnmanagedType.LPTStr)] public string title;
        public uint flags;
        public BrowseCallbackProc callback;
        public IntPtr callbackData;
        public int image;
    }

    private static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
    private const uint BIF_RETURNONLYFSDIRS = 0x0001;
    private const uint BIF_EDITBOX = 0x0010;
    private const uint BIF_NEWDIALOGSTYLE = 0x0040;
    private const uint BFFM_INITIALIZED = 1;
    private const uint SWP_NOSIZE = 0x0001;
    private const uint SWP_NOMOVE = 0x0002;

    [DllImport("shell32.dll", CharSet = CharSet.Auto)]
    private static extern IntPtr SHBrowseForFolder(ref BrowseInfo info);
    [DllImport("shell32.dll", CharSet = CharSet.Auto)]
    private static extern bool SHGetPathFromIDList(IntPtr pidl, StringBuilder path);
    [DllImport("ole32.dll")]
    private static extern void CoTaskMemFree(IntPtr memory);
    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SetWindowPos(
        IntPtr hwnd, IntPtr insertAfter, int x, int y, int width, int height, uint flags);
    [DllImport("user32.dll")]
    private static extern bool BringWindowToTop(IntPtr hwnd);
    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hwnd);

    public static string Show(IntPtr owner, string title) {
        BrowseCallbackProc callback = delegate(IntPtr hwnd, uint message, IntPtr lParam, IntPtr data) {
            if (message == BFFM_INITIALIZED) {
                SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
                BringWindowToTop(hwnd);
                SetForegroundWindow(hwnd);
            }
            return 0;
        };
        BrowseInfo info = new BrowseInfo();
        info.owner = owner;
        info.title = title;
        info.flags = BIF_RETURNONLYFSDIRS | BIF_EDITBOX | BIF_NEWDIALOGSTYLE;
        info.callback = callback;
        IntPtr pidl = SHBrowseForFolder(ref info);
        if (pidl == IntPtr.Zero) return null;
        try {
            StringBuilder path = new StringBuilder(32768);
            return SHGetPathFromIDList(pidl, path) ? path.ToString() : null;
        } finally {
            CoTaskMemFree(pidl);
            GC.KeepAlive(callback);
        }
    }
}
'@
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::None
$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
$owner.Location = New-Object System.Drawing.Point(-32000, -32000)
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.Opacity = 0.01
$owner.Show()
$owner.Activate()
$dialogProcessId = [LvatsDialogTopmost]::CurrentProcessId()
[LvatsDialogTopmost]::Start($dialogProcessId, $owner.Handle)
"""

# 文件框由独立监视线程捕获；文件夹框在 Shell 初始化回调中直接设置 HWND_TOPMOST；
# finally 里 Close + Dispose，超时/异常都不残留窗口。
PICK_PS1 = PS_OWNER_PREAMBLE + r"""
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.Multiselect = $true
$dlg.Title = '选择要转录的音频 / 视频文件（可多选）'
$dlg.Filter = '音频/视频|*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.aac;*.mp4;*.mov;*.mkv;*.webm;*.avi;*.ts;*.m4v|所有文件|*.*'
try {
    $res = $dlg.ShowDialog($owner)
    if ($res -eq [System.Windows.Forms.DialogResult]::OK) {
        $dlg.FileNames | ForEach-Object { Write-Output $_ }
    }
} finally {
    [LvatsDialogTopmost]::Stop()
    $owner.Close()
    $owner.Dispose()
}
"""

FOLDER_PS1 = PS_OWNER_PREAMBLE + r"""
try {
    $selected = [LvatsFolderPicker]::Show(
        $owner.Handle, '选择要转录的文件夹（递归扫描音频/视频）')
    if ($selected) {
        Write-Output $selected
    }
} finally {
    [LvatsDialogTopmost]::Stop()
    $owner.Close()
    $owner.Dispose()
}
"""

PICK_JOBS: dict[str, dict] = {}


def _scan_folder(folder: str, limit: int = 2000) -> list[str]:
    out: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for fn in sorted(files):
            if Path(fn).suffix.lower() in ALLOWED_SUFFIXES:
                out.append(os.path.join(root, fn))
                if len(out) >= limit:
                    return out
    return out


def _run_pick(pick_id: str, kind: str):
    rec = PICK_JOBS[pick_id]
    try:
        ps1 = PICK_PS1 if kind == "files" else FOLDER_PS1
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", ps1],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()[:300]
            rec.update(status="error", paths=[],
                       error=f"原生对话框不可用（{err or '未知错误'}）。服务若跑在后台/非交互会话会弹不出窗口，请改用粘贴框添加。")
            return
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        paths = lines if kind == "files" else (_scan_folder(lines[0]) if lines else [])
        rec.update(status="done", paths=paths, error=None)
    except subprocess.TimeoutExpired:
        rec.update(status="error", paths=[], error="对话框 120 秒未操作，已超时。可用粘贴框添加。")
    except Exception as e:
        rec.update(status="error", paths=[], error=f"调起对话框失败：{e}。可用粘贴框添加。")


def _start_pick(kind: str) -> str:
    pick_id = uuid.uuid4().hex[:10]
    PICK_JOBS[pick_id] = {"status": "pending", "paths": [], "error": None}
    threading.Thread(target=_run_pick, args=(pick_id, kind), daemon=True,
                     name=f"pick-{kind}").start()
    return pick_id


@app.post("/api/pick-files")
async def api_pick_files():
    return {"ok": True, "pick_id": _start_pick("files")}


@app.post("/api/pick-folder")
async def api_pick_folder():
    return {"ok": True, "pick_id": _start_pick("folder")}


@app.get("/api/pick-result")
async def api_pick_result(pick_id: str = Query(...)):
    rec = PICK_JOBS.get(pick_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="pick 任务不存在")
    return {"pick_id": pick_id, **rec}


# ──────────────────────────────────────────────────────────────
# 文件队列
# ──────────────────────────────────────────────────────────────
@app.post("/api/files/add")
async def api_files_add(payload: dict):
    paths = payload.get("paths") or []
    added, skipped, limited = [], 0, 0
    with _files_lock:
        known_paths = {f["path"] for f in FILES.values()}
        for raw in paths:
            p = _norm(raw)
            if not p:
                continue
            if p in known_paths:
                skipped += 1
                continue
            if len(FILES) + _drop_reservations >= MAX_FILES:
                limited += 1
                continue
            ok, reason = _check_path(p)
            fid = uuid.uuid4().hex[:10]
            FILES[fid] = {"id": fid, "path": p, "name": Path(p).name, "ok": ok, "reason": reason}
            known_paths.add(p)
            added.append(FILES[fid])
    return {"ok": True, "added": added, "skipped": skipped,
            "limited": limited, "limit": MAX_FILES}


@app.post("/api/files/drop")
async def api_file_drop(request: Request, name: str = Query(..., min_length=1, max_length=255)):
    """接收浏览器拖入的单个文件，流式保存为 Lvats 管理的本机临时副本。"""
    global _drop_reservations
    origin = (request.headers.get("origin") or "").rstrip("/")
    allowed_origins = {f"{SERVICE_SCHEME}://127.0.0.1:{PORT}",
                       f"{SERVICE_SCHEME}://localhost:{PORT}"}
    if origin and origin not in allowed_origins:
        raise HTTPException(status_code=403, detail="仅允许从 Lvats 本机页面拖放文件")

    original_name = sanitize_filename(Path(name).name)
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix or '无扩展名'}")

    with _files_lock:
        if len(FILES) + _drop_reservations >= MAX_FILES:
            raise HTTPException(status_code=409, detail=f"文件队列最多 {MAX_FILES} 个，请先移除文件")
        _drop_reservations += 1

    token = uuid.uuid4().hex
    partial = UPLOAD_DIR / f".{token}.part"
    target = UPLOAD_DIR / f"{token[:12]}_{original_name}"
    written = 0
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        try:
            expected_size = int(request.headers.get("content-length") or 0)
        except ValueError:
            expected_size = 0
        if expected_size < 0:
            raise HTTPException(status_code=400, detail="文件大小无效")
        if expected_size and expected_size + 64 * 1024 * 1024 > shutil.disk_usage(UPLOAD_DIR).free:
            raise HTTPException(status_code=507, detail="磁盘剩余空间不足，无法保存拖放临时副本")
        with partial.open("xb") as output:
            async for chunk in request.stream():
                if chunk:
                    await asyncio.to_thread(output.write, chunk)
                    written += len(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="不能添加空文件")
        os.replace(partial, target)
        fid = uuid.uuid4().hex[:10]
        record = {
            "id": fid, "path": _norm(target), "name": original_name,
            "ok": True, "reason": "", "managed_upload": True, "upload_size": written,
        }
        with _files_lock:
            FILES[fid] = record
        _log("t", f"拖放上传完成：{original_name}（{written} 字节）")
        return {"ok": True, "file": record}
    except HTTPException:
        partial.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"保存拖放临时副本失败：{exc}") from exc
    finally:
        with _files_lock:
            _drop_reservations -= 1


@app.post("/api/files/remove")
async def api_files_remove(payload: dict):
    fid = payload.get("file_id") or ""
    with _files_lock:
        f = FILES.pop(fid, None)
    if f is None:
        raise HTTPException(status_code=404, detail="文件不在队列中")
    removed = 0
    for tid in [tid for tid, t in TASKS.items()
                if t["path"] == f["path"] and t["status"] == "queued"]:
        TASKS.pop(tid)
        if TASK_QUEUE:
            TASK_QUEUE.remove(tid)
        removed += 1
    cleaned = _cleanup_managed_if_unused(f["path"], bool(f.get("managed_upload")))
    mark_dirty()
    return {"ok": True, "removed_tasks": removed, "upload_cleaned": cleaned}


# ──────────────────────────────────────────────────────────────
# 任务队列
# ──────────────────────────────────────────────────────────────
@app.post("/api/tasks/add")
async def api_tasks_add(payload: dict):
    fids = payload.get("file_ids") or []
    fmt = (payload.get("format") or "").lower().strip()
    prompt = (payload.get("prompt") or "").strip()
    model_id = (payload.get("model_id") or "").strip() or None   # v0.6.0：任务级模型选择
    if fmt not in {"txt", "srt", "vtt"}:
        raise HTTPException(status_code=400, detail="format 只能是 txt / srt / vtt")
    if model_id and not model_registry.get(model_id):
        raise HTTPException(status_code=400, detail=f"模型 {model_id} 不在注册表中，请先重新扫描或下载")
    if not model_id:
        model_id = (model_registry.default_entry() or {}).get('id')

    created = skipped = 0
    for fid in fids:
        f = FILES.get(fid)
        if f is None or not f["ok"]:
            continue
        dup = any(
            t["path"] == f["path"] and t["format"] == fmt and t["status"] in ("queued", "running")
            for t in TASKS.values()
        )
        if dup:
            skipped += 1
            continue
        tid = uuid.uuid4().hex[:10]
        TASKS[tid] = {
            "id": tid, "path": f["path"], "name": f["name"], "format": fmt,
            "prompt": prompt,
            "status": "queued", "error": None, "progress": 0.0, "message": "",
            "partial_text": "", "warning": None,
            "output_path": None, "output_name": None, "full_text": "",
            "duration": 0.0, "cancel_requested": False, "model_id": model_id,
            "managed_upload": bool(f.get("managed_upload")),
            "archived": False, "archived_at": None,
            "created_at": time.time(), "started_at": None, "finished_at": None,
        }
        TASK_QUEUE.put(tid)
        created += 1
        _log("t", f"＋ 入队：{f['name']} [{fmt}]", task_id=tid)
    mark_dirty()
    persist_now()
    return {"ok": True, "created": created, "skipped": skipped}


@app.delete("/api/tasks/{task_id}")
async def api_task_delete(task_id: str):
    """
    删除任务。
      排队中  → 直接从队列移除
      完成/失败/已取消 → 只移除队列行，不删 output/ 中已生成的文件
      运行中  → 拒绝（请先取消）
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if t["status"] == "running":
        raise HTTPException(status_code=409, detail="任务正在运行中，请先取消")

    TASKS.pop(task_id, None)
    if TASK_QUEUE:
        TASK_QUEUE.remove(task_id)
    _log("t", f"－ 删除：{t['name']} [{t['format']}]", task_id=task_id)
    _cleanup_managed_if_unused(t["path"], bool(t.get("managed_upload")))
    mark_dirty()
    persist_now()
    return {"ok": True, "deleted": task_id}


def _archive_completed_task(t: dict) -> bool:
    """归档完成任务；保留任务记录与 output/ 产物，仅改变展示分组。"""
    if t.get("status") != "done" or t.get("archived"):
        return False
    t["archived"] = True
    t["archived_at"] = time.time()
    return True


@app.post("/api/tasks/{task_id}/archive")
async def api_task_archive(task_id: str):
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if t.get("status") != "done":
        raise HTTPException(status_code=409, detail="只有已完成任务可以归档")
    if t.get("archived"):
        raise HTTPException(status_code=409, detail="任务已经归档")
    _archive_completed_task(t)
    _log("t", f"▣ 归档：{t['name']} [{t['format']}]", task_id=task_id)
    mark_dirty()
    persist_now()
    return {"ok": True, "archived": task_id}


@app.post("/api/tasks/archive-completed")
async def api_tasks_archive_completed():
    archived = []
    for t in TASKS.values():
        if _archive_completed_task(t):
            archived.append(t["id"])
    if archived:
        _log("t", f"▣ 已归档 {len(archived)} 条完成任务")
        mark_dirty()
        persist_now()
    return {"ok": True, "archived": archived, "count": len(archived)}


@app.post("/api/tasks/{task_id}/restore")
async def api_task_restore(task_id: str):
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not t.get("archived"):
        raise HTTPException(status_code=409, detail="任务未归档")
    t["archived"] = False
    t["archived_at"] = None
    _log("t", f"□ 恢复归档：{t['name']} [{t['format']}]", task_id=task_id)
    mark_dirty()
    persist_now()
    return {"ok": True, "restored": task_id}


@app.post("/api/tasks/{task_id}/cancel")
async def api_task_cancel(task_id: str):
    """
    协作式取消：只置标志，由推理循环在最近的切块边界自行停止（延迟 ≤ 一个切块）。
    取消后不写产物文件，已识别文本保留在 partial_text 供页面查看。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if t["status"] != "running":
        raise HTTPException(status_code=409, detail="只有运行中的任务可以取消")
    t["cancel_requested"] = True
    tok = t.get("_token")
    if tok:
        tok.set()
    _log("warn", f"■ 取消请求：{t['name']}（将在最近一个切块边界停止）", task_id=task_id)
    mark_dirty()
    return {"ok": True, "cancelling": task_id}


@app.post("/api/tasks/{task_id}/move")
async def api_task_move(task_id: str, payload: dict):
    """
    上下挪动，仅对「排队中」任务生效。delta: -1 上移 / +1 下移。
    running 任务已出队，因此不可能被越过去。
    """
    t = TASKS.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if t["status"] != "queued":
        raise HTTPException(status_code=409, detail="只有排队中的任务可以挪动")
    try:
        delta = int(payload.get("delta", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="delta 必须是 -1 或 1")
    if delta not in (-1, 1):
        raise HTTPException(status_code=400, detail="delta 必须是 -1 或 1")

    if not TASK_QUEUE.move(task_id, delta):
        raise HTTPException(status_code=400, detail="已在边界，无法继续挪动")
    mark_dirty()
    persist_now()
    return {"ok": True, "order": TASK_QUEUE.snapshot()}


# ──────────────────────────────────────────────────────────────
# 状态快照
# ──────────────────────────────────────────────────────────────
def _ordered_tasks() -> list[dict]:
    """未归档：运行中→排队中→已结束；归档任务最后按归档时间倒序。"""
    order = TASK_QUEUE.snapshot() if TASK_QUEUE else []
    pos = {tid: i for i, tid in enumerate(order)}

    def key(t):
        if t.get("archived"):
            return (3, -(t.get("archived_at") or t.get("finished_at") or 0))
        if t["status"] == "running":
            return (0, 0)
        if t["status"] == "queued":
            return (1, pos.get(t["id"], 1_000_000))
        return (2, -(t.get("finished_at") or 0))

    return sorted(TASKS.values(), key=key)


@app.get("/api/state")
async def api_state():
    done_by_path: dict[str, list[dict]] = {}
    for t in TASKS.values():
        if t["status"] == "done":
            done_by_path.setdefault(t["path"], []).append(t)

    files = []
    for f in FILES.values():
        latest: dict[str, dict] = {}
        for t in done_by_path.get(f["path"], []):
            cur = latest.get(t["format"])
            if cur is None or (t["finished_at"] or 0) > (cur["finished_at"] or 0):
                latest[t["format"]] = t
        files.append({**f, "done_formats": {fmt: latest[fmt]["id"] for fmt in sorted(latest)}})

    tasks = []
    for t in _ordered_tasks():
        el_text, _ = _task_elapsed(t)
        m_entry = model_registry.get(t.get("model_id")) or model_registry.default_entry()
        tasks.append({
            "id": t["id"], "path": t["path"], "name": t["name"], "format": t["format"],
            "status": t["status"], "error": t["error"], "progress": t["progress"],
            "elapsed": el_text, "partial_text": t["partial_text"],
            "warning": t["warning"], "output_name": t["output_name"],
            "message": t.get("message") or "",
            "cancel_requested": bool(t.get("cancel_requested")),
            "model_id": t.get("model_id"),
            "model_label": (m_entry or {}).get("label", "—"),
            "archived": bool(t.get("archived")),
            "archived_at": t.get("archived_at"),
        })
    return {"files": files, "tasks": tasks,
            "order": TASK_QUEUE.snapshot() if TASK_QUEUE else [],
            "models": model_registry.all_models(),
            "version": VERSION}


@app.get("/api/read-result")
async def api_read_result(task_id: str = Query(...)):
    t = TASKS.get(task_id)
    if t is None or t["status"] != "done" or not t["output_path"]:
        raise HTTPException(status_code=404, detail="结果不可用")
    if not Path(t["output_path"]).exists():
        raise HTTPException(status_code=404, detail="输出文件已不存在")
    return FileResponse(t["output_path"], filename=t["output_name"],
                        media_type="application/octet-stream")


# ──────────────────────────────────────────────────────────────
# SSE：日志 / 实时文本 / 耗时 / 设备（合并为一条流）
# ──────────────────────────────────────────────────────────────
@app.get("/api/events")
async def api_events(request: Request, last_event_id: str | None = None, stream_id: str | None = None):
    q = BUS.subscribe()
    try:
        last = max(0, int(request.headers.get('last-event-id') or last_event_id or 0))
    except ValueError:
        last = 0
    # 服务重启后序号归零，旧游标不能阻止新会话事件发送。
    if last > BUS._seq or (stream_id is not None and stream_id != BUS.stream_id):
        last = 0

    def sse(ev: dict) -> str:
        return (f"id: {ev['seq']}\nevent: {ev['kind']}\n"
                f"data: {json.dumps(ev, ensure_ascii=False)}\n\n")

    async def gen():
        cursor = last
        try:
            yield "retry: 2000\n\n"
            yield f"event: stream\ndata: {json.dumps({'stream_id': BUS.stream_id})}\n\n"
            for ev in BUS.since(cursor):
                yield sse(ev)
                cursor = ev['seq']
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if ev['seq'] > cursor:
                    yield sse(ev)
                    cursor = ev['seq']
        finally:
            BUS.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


# ──────────────────────────────────────────────────────────────
# 快捷提示词持久化
# ──────────────────────────────────────────────────────────────
def _load_prompts() -> list[dict]:
    if not QUICK_PROMPTS_FILE.exists():
        return []
    try:
        data = json.loads(QUICK_PROMPTS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_prompts(prompts: list[dict]):
    QUICK_PROMPTS_FILE.write_text(json.dumps(prompts, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


@app.get("/api/prompts")
async def api_prompts_list():
    return {"prompts": _load_prompts()}


@app.post("/api/prompts")
async def api_prompts_create(item: dict):
    name = (item.get("name") or "").strip()
    content = (item.get("content") or "").strip()
    color = item.get("color") or {"r": 255, "g": 160, "b": 60}
    if not name:
        raise HTTPException(status_code=400, detail="名称不能为空")
    if len(name) > 30:
        raise HTTPException(status_code=400, detail="名称不能超过 30 字符")
    if not content:
        raise HTTPException(status_code=400, detail="提示词内容不能为空")
    prompts = _load_prompts()
    if any(p["name"] == name for p in prompts):
        raise HTTPException(status_code=400, detail="名称已存在")
    if len(prompts) >= MAX_QUICK_PROMPTS:
        raise HTTPException(status_code=400, detail=f"快捷提示最多保存 {MAX_QUICK_PROMPTS} 条")
    now = datetime.datetime.now().isoformat()
    prompts.append({"id": uuid.uuid4().hex[:12], "name": name, "content": content,
                    "color": color, "created_at": now, "updated_at": now})
    _save_prompts(prompts)
    return {"ok": True, "prompt": prompts[-1]}


@app.put("/api/prompts/{prompt_id}")
async def api_prompts_update(prompt_id: str, item: dict):
    prompts = _load_prompts()
    idx = next((i for i, p in enumerate(prompts) if p.get("id") == prompt_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="快捷提示不存在")
    name = (item.get("name") or "").strip()
    color = item.get("color")
    if name:
        if len(name) > 30:
            raise HTTPException(status_code=400, detail="名称不能超过 30 字符")
        if any(p["name"] == name and p.get("id") != prompt_id for p in prompts):
            raise HTTPException(status_code=400, detail="名称已存在")
        prompts[idx]["name"] = name
    if color is not None:
        prompts[idx]["color"] = color
    if "content" in item:
        prompts[idx]["content"] = item["content"]
    prompts[idx]["updated_at"] = datetime.datetime.now().isoformat()
    _save_prompts(prompts)
    return {"ok": True, "prompt": prompts[idx]}


@app.delete("/api/prompts/{prompt_id}")
async def api_prompts_delete(prompt_id: str):
    prompts = _load_prompts()
    new_prompts = [p for p in prompts if p.get("id") != prompt_id]
    if len(new_prompts) == len(prompts):
        raise HTTPException(status_code=404, detail="快捷提示不存在")
    _save_prompts(new_prompts)
    return {"ok": True}


# ──────────────────────────────────────────────────────────────
# 服务停止 / 重启（停止 v0.5.3，重启 v0.9.2）
# ──────────────────────────────────────────────────────────────
_shutdown_started = False


def _write_restart_status(token: str, phase: str, message: str, **extra):
    """原子写入跨进程重启状态；新服务启动后仍可向原页面报告结果。"""
    payload = {
        "token": token,
        "phase": phase,
        "message": message,
        "updated_at": time.time(),
        **extra,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESTART_STATUS_FILE.with_suffix(RESTART_STATUS_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, RESTART_STATUS_FILE)


def _read_restart_status() -> dict:
    if not RESTART_STATUS_FILE.is_file():
        return {"phase": "idle", "message": "当前没有重启任务"}
    try:
        data = json.loads(RESTART_STATUS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"phase": "failed", "message": "重启状态文件格式无效"}
    except Exception as e:
        return {"phase": "failed", "message": f"无法读取重启状态：{e}"}


def _restart_preflight():
    if os.name != "nt":
        raise RuntimeError("自动重启目前只支持 Windows 本机服务")
    if not RESTART_HELPER.is_file():
        raise RuntimeError(f"缺少重启辅助程序：{RESTART_HELPER}")
    if not Path(sys.executable).is_file():
        raise RuntimeError(f"当前运行程序不存在：{sys.executable}")
    if not TEST_MODE:
        starter = BASE_DIR / "scripts" / "start_lvats.ps1"
        powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        if not starter.is_file():
            raise RuntimeError(f"缺少 Windows 启动脚本：{starter}")
        if not powershell.is_file():
            raise RuntimeError(f"找不到 Windows PowerShell：{powershell}")


def _spawn_restart_helper(token: str) -> int:
    command = [
        sys.executable, str(RESTART_HELPER),
        "--project-root", str(BASE_DIR),
        "--port", str(PORT),
        "--old-pid", str(os.getpid()),
        "--token", token,
        "--expected-version", VERSION,
        "--status-file", str(RESTART_STATUS_FILE),
        "--scheme", SERVICE_SCHEME,
    ]
    if TEST_MODE:
        command.append("--test-mode")
        test_command = os.environ.get("ASR_RESTART_TEST_COMMAND", "").strip()
        if test_command:
            command.extend(["--test-command-json", test_command])
    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
             | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
             | getattr(subprocess, "DETACHED_PROCESS", 0))
    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=flags,
    )
    return process.pid


def _interrupt_running_tasks() -> int:
    interrupted = 0
    for task in TASKS.values():
        if task["status"] == "running":
            task["cancel_requested"] = True
            token = task.get("_token")
            if token:
                token.set()
            interrupted += 1
    return interrupted


def _schedule_process_exit(delay: float = 2.0):
    def _do_exit():
        persist_now()  # 二次落盘；首次严格落盘已在重启辅助进程启动前完成
        time.sleep(delay)  # 给响应留出回到前端的时间
        _remove_owned_pid_file()
        os._exit(0)

    threading.Thread(target=_do_exit, name="lvats-exit", daemon=True).start()


@app.get("/api/restart/status")
async def api_restart_status():
    return _read_restart_status()


@app.post("/api/restart")
async def api_restart():
    """保存队列后退出，由独立辅助进程等待端口释放并复用正式启动链路。"""
    global _shutdown_started
    if _shutdown_started:
        raise HTTPException(status_code=409, detail="停止或重启流程已在进行中")

    token = uuid.uuid4().hex
    try:
        _restart_preflight()
        persist_now(strict=True)
        _write_restart_status(token, "scheduled", "队列已安全保存，正在安排重启", old_pid=os.getpid())
        helper_pid = _spawn_restart_helper(token)
    except Exception as e:
        try:
            _write_restart_status(token, "failed", f"重启未开始：{e}", old_pid=os.getpid())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"重启未开始：{e}") from e

    _shutdown_started = True
    interrupted = _interrupt_running_tasks()
    _log("warn", f"[{SERVICE_NAME}] 收到重启请求：队列已保存，"
                 f"{interrupted} 个运行中任务将在新进程中重新排队。")
    _schedule_process_exit()
    return {
        "ok": True,
        "token": token,
        "old_pid": os.getpid(),
        "helper_pid": helper_pid,
        "interrupted": interrupted,
        "message": "队列已保存，服务正在重启",
    }


@app.post("/api/shutdown")
async def api_shutdown():
    """
    停止整个 Lvats 服务：
      1. 响应先回前端；
      2. 运行中任务置协作式取消标志（切块边界停止；若进程先退出，
         队列持久化里 running 记录会在下次启动自动重置为 queued 重跑）；
      3. 落盘队列 → 删 pid 文件 → 进程几秒内完全退出（端口随之释放）。
    """
    global _shutdown_started
    if _shutdown_started:
        return {"ok": True, "message": "停止流程已在进行中"}
    _shutdown_started = True
    interrupted = _interrupt_running_tasks()
    _log("warn", f"[{SERVICE_NAME}] 收到停止请求："
                 f"{interrupted} 个运行中任务协作式中断，进程即将退出。"
                 f"重启请双击 启动Lvats.bat")

    _schedule_process_exit()
    return {"ok": True, "interrupted": interrupted,
            "message": "服务将在几秒内停止；如需重启请双击 启动Lvats.bat"}


# ──────────────────────────────────────────────────────────────
# 模型控制 / 健康
# ──────────────────────────────────────────────────────────────
@app.post("/api/unload")
async def api_unload():
    if _models_busy():
        raise HTTPException(status_code=409, detail="任务或预加载正在执行，请完成后再卸载模型。")
    await asyncio.to_thread(_unload_models)
    _log("t", "模型已卸载，显存释放")
    return {"ok": True}


@app.post("/api/preload")
async def api_preload(payload: dict | None = None):
    global _preload_task
    model_id = (payload or {}).get('model_id')
    entry = model_registry.get(model_id) if model_id else model_registry.default_entry()
    if entry is None:
        raise HTTPException(status_code=400, detail="所选模型不可用，请先重新扫描模型。")
    if _models_busy():
        return {'ok': True, 'skipped': True, 'reason': '任务或预加载正在执行'}
    def _do():
        try:
            with asr_model.INFERENCE_LOCK:
                if entry['engine'] == 'faster-whisper':
                    asr_model.unload_model()
                    fw_engine.preload(entry, _preload_cb)
                else:
                    fw_engine.unload()
                    asr_model.preload(entry['path'])
            _log("t", f"模型预加载完成：{entry['label']}" +
                 ('（占位模式，未加载权重）' if asr_model.USE_PLACEHOLDER else ''))
        except Exception as e:
            _log("err", f"模型预加载失败：{e}")
    _preload_task = asyncio.create_task(asyncio.to_thread(_do), name='model-preload')
    return {"ok": True, 'model_id': entry['id']}


def _preload_cb(pct, msg, partial=None, meta=None):
    meta = meta or {}
    _log(meta.get('level', 't'), meta.get('log') or msg)


def _models_busy():
    return (any(t['status'] in ('running', 'queued') for t in TASKS.values()) or
            (_preload_task is not None and not _preload_task.done()))


def _unload_models():
    with asr_model.INFERENCE_LOCK:
        asr_model.unload_model()
        fw_engine.unload()


def _unload_idle_models():
    if not asr_model.INFERENCE_LOCK.acquire(blocking=False):
        return
    try:
        for engine, loaded, elapsed, unload in (
            ('Qwen', asr_model.is_loaded, asr_model.seconds_since_last_use, asr_model.unload_model),
            ('faster-whisper', fw_engine.is_loaded, fw_engine.seconds_since_last_use, fw_engine.unload),
        ):
            if loaded() and elapsed() >= asr_model.IDLE_TIMEOUT_SECONDS:
                unload()
                _log('t', f'{engine} 闲置超时，已释放模型')
    finally:
        asr_model.INFERENCE_LOCK.release()


async def _idle_models_loop():
    while True:
        await asyncio.sleep(15)
        await asyncio.to_thread(_unload_idle_models)


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@app.get("/api/health")
async def api_health():
    info = fw_engine.get_device_info() if fw_engine.is_loaded() else asr_model.get_device_info()
    base = {"name": SERVICE_NAME, "service": SERVICE_FULLNAME, "version": VERSION,
            "pid": os.getpid(),
            "https": HTTPS_ENABLED, "url": SERVICE_URL,
            "http_redirect": HTTP_REDIRECT_ENABLED,
            "model_loaded": asr_model.is_loaded() or fw_engine.is_loaded(),
            "aligner_available": asr_model.aligner_available(),
            "placeholder": asr_model.USE_PLACEHOLDER,
            "test_mode": TEST_MODE,
            "device_info": info,
            "wav_cache": asr_model.wav_cache.stats()}
    if not _torch_ok():
        return {**base, "cuda": False, "gpu_name": None}
    import torch
    cuda = torch.cuda.is_available()
    return {**base, "cuda": cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None}


# ──────────────────────────────────────────────────────────
# 模型管理（v0.6.0）：列表 / 重新扫描 / HF 下载
# ──────────────────────────────────────────────────────────
DOWNLOAD_JOBS: dict[str, dict] = {}   # did -> {name, repo_id, status, error, note, mb}


def _queue_download(repo_id: str, name: str, *, note: str = "",
                    start_immediately: bool = True) -> str | None:
    if model_registry.get(name):
        return None
    for did, job in DOWNLOAD_JOBS.items():
        if job["status"] in ("queued", "running") and job["name"] == name:
            return None
    did = uuid.uuid4().hex[:10]
    DOWNLOAD_JOBS[did] = {"name": name, "repo_id": repo_id,
                          "status": "running" if start_immediately else "queued",
                          "error": None, "note": note, "mb": 0.0}
    if start_immediately:
        threading.Thread(target=_download_worker, args=(did, repo_id, name),
                         name=f"dl-{name}", daemon=True).start()
    _log("t", f"[模型下载] 已排队：{repo_id} → models/{name}")
    return did


def _queue_initial_model_downloads():
    downloads = [*model_registry.BUILTIN_DOWNLOADS, model_registry.ALIGNER_DOWNLOAD]
    queued = []
    for item in downloads:
        name = item["repo_id"].split("/")[-1]
        if name == model_registry.ALIGNER_DOWNLOAD["repo_id"].split("/")[-1]:
            target = model_registry.MODELS_DIR / name
            if asr_model.find_aligner_dir() is not None:
                continue
            if target.exists() and (target / "model.safetensors").is_file():
                continue
        did = _queue_download(item["repo_id"], name, note="首次启动自动下载",
                              start_immediately=False)
        if did:
            queued.append((did, item["repo_id"], name))
    if queued:
        def _download_batch():
            for did, repo_id, name in queued:
                _download_worker(did, repo_id, name)
        threading.Thread(target=_download_batch, name="dl-initial-models", daemon=True).start()
        _log("t", f"[模型下载] 首次启动将在后台依次补齐 {len(queued)} 个模型；服务界面可正常使用")


def _rescan_and_publish():
    reg = model_registry.rescan()
    BUS.publish("models", models=list(reg.values()))
    return reg


@app.get("/api/models")
async def api_models():
    return {"models": model_registry.all_models(),
            "builtin": model_registry.BUILTIN_DOWNLOADS,
            "downloads": list(DOWNLOAD_JOBS.values())}


@app.post("/api/models/rescan")
async def api_models_rescan():
    reg = _rescan_and_publish()
    _log("ok", f"[多模型] 重新扫描完成：{len(reg)} 个模型（"
               f"{', '.join(m['label'] for m in reg.values()) or '无'}）")
    return {"ok": True, "models": list(reg.values())}


def _download_worker(did: str, repo_id: str, name: str):
    job = DOWNLOAD_JOBS[did]
    job["status"] = "running"
    target = model_registry.MODELS_DIR / name

    def _poll_size():
        while job["status"] == "running":
            try:
                mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
                job["mb"] = round(mb, 1)
                BUS.publish("download", did=did, name=name, repo_id=repo_id,
                            status="running", mb=job["mb"])
            except Exception:
                pass
            time.sleep(2)

    threading.Thread(target=_poll_size, name=f"dl-poll-{name}", daemon=True).start()
    try:
        from huggingface_hub import snapshot_download
        _log("t", f"[模型下载] 开始 {repo_id} → models/{name}"
                   f"（镜像：{os.environ.get('HF_ENDPOINT') or '官方源'}）")
        snapshot_download(repo_id=repo_id, local_dir=str(target))
        job["status"] = "done"
        _rescan_and_publish()
        BUS.publish("download", did=did, name=name, repo_id=repo_id, status="done")
        _log("ok", f"[模型下载] 完成 {name}，已注册，立即可选")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)[:300]
        BUS.publish("download", did=did, name=name, repo_id=repo_id,
                    status="error", error=job["error"])
        _log("err", f"[模型下载] 失败 {repo_id}：{e}（半成品目录已保留 models/{name}，可重试）")


@app.post("/api/models/download")
async def api_models_download(payload: dict):
    repo_id = (payload.get("repo_id") or "").strip()
    name = (payload.get("name") or "").strip() or (repo_id.split("/")[-1] if "/" in repo_id else "")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise HTTPException(status_code=400, detail="repo_id 需为 '组织/模型名' 格式，如 Systran/faster-whisper-tiny")
    if not name or name != sanitize_filename(name):
        raise HTTPException(status_code=400, detail=f"目录名非法：{name}")
    if (model_registry.MODELS_DIR / name).exists() and model_registry.get(name):
        raise HTTPException(status_code=409, detail=f"models/{name} 已存在且已注册，无需下载")
    did = _queue_download(repo_id, name)
    if did is None:
        raise HTTPException(status_code=409, detail=f"models/{name} 已存在且已注册，无需下载")
    return {"ok": True, "download_id": did, "name": name}


@app.get("/api/models/download/status")
async def api_models_download_status():
    return {"downloads": list(DOWNLOAD_JOBS.values())}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


def _http_redirect_response(request_line: bytes) -> bytes:
    """Build a same-host HTTPS redirect for an origin-form HTTP request target."""
    try:
        parts = request_line.decode("latin-1", errors="replace").strip().split(" ", 2)
        target = parts[1] if len(parts) == 3 and parts[1].startswith("/") else "/"
    except Exception:
        target = "/"
    location = f"{SERVICE_URL}{target}"
    return (
        "HTTP/1.1 308 Permanent Redirect\r\n"
        f"Location: {location}\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-store\r\n\r\n"
    ).encode("latin-1")


async def _relay_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    while data := await reader.read(64 * 1024):
        writer.write(data)
        await writer.drain()


async def _handle_public_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """TLS records are tunneled to Uvicorn; plaintext HTTP receives a 308 redirect."""
    backend_writer = None
    relays = []
    try:
        first = await asyncio.wait_for(reader.read(1), timeout=10)
        if not first:
            return
        if first != b"\x16":
            request_line = first + await asyncio.wait_for(reader.readline(), timeout=5)
            writer.write(_http_redirect_response(request_line))
            await writer.drain()
            return
        backend_reader, backend_writer = await asyncio.open_connection("127.0.0.1", TLS_BACKEND_PORT)
        backend_writer.write(first)
        await backend_writer.drain()
        relays = [asyncio.create_task(_relay_stream(reader, backend_writer)),
                  asyncio.create_task(_relay_stream(backend_reader, writer))]
        _, pending = await asyncio.wait(relays, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*relays, return_exceptions=True)
    except (ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError):
        pass
    finally:
        for stream in (backend_writer, writer):
            if stream is not None:
                stream.close()
                try:
                    await stream.wait_closed()
                except Exception:
                    pass


async def _serve_https_with_redirect(uvicorn_module):
    """Keep HTTPS on the public port while accepting and redirecting plaintext HTTP there too."""
    config = uvicorn_module.Config(app, **_uvicorn_kwargs(TLS_BACKEND_PORT))
    backend = uvicorn_module.Server(config)
    backend.install_signal_handlers = lambda: None
    backend_task = asyncio.create_task(backend.serve(), name="lvats-tls-backend")
    proxy = None
    proxy_task = None
    try:
        for _ in range(300):
            if backend.started:
                break
            if backend_task.done():
                await backend_task
                raise RuntimeError(f"HTTPS 内部服务未能启动（端口 {TLS_BACKEND_PORT}）")
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError(f"HTTPS 内部服务启动超时（端口 {TLS_BACKEND_PORT}）")
        proxy = await asyncio.start_server(_handle_public_connection, "127.0.0.1", PORT)
        _log("ok", f"[HTTP] http://127.0.0.1:{PORT} → {SERVICE_URL}（308）")
        proxy_task = asyncio.create_task(proxy.serve_forever(), name="lvats-public-gateway")
        done, pending = await asyncio.wait((backend_task, proxy_task), return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            error = task.exception()
            if error:
                raise error
    finally:
        if proxy is not None:
            proxy.close()
            await proxy.wait_closed()
        backend.should_exit = True
        if not backend_task.done():
            await backend_task


if __name__ == "__main__":
    import uvicorn
    _set_console_title(SERVICE_NAME)
    print(f"[{SERVICE_NAME}] v{VERSION} 启动：{SERVICE_URL}", flush=True)
    if HTTPS_ENABLED and HTTP_REDIRECT_ENABLED:
        asyncio.run(_serve_https_with_redirect(uvicorn))
    else:
        uvicorn.run(app, **_uvicorn_kwargs())
