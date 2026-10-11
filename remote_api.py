# -*- coding: utf-8 -*-
"""Opt-in OpenAI-compatible audio transcription; no model imports or fallback.

Server integration:
    store = ConfigStore(lvats_paths.CONFIG_DIR / "remote_api.json")
    # Persist only store.public()["revision"] with a queued task, never runtime().
    config = store.runtime(expected_revision=task["remote_revision"])
    result = await transcribe(path, config, need_timestamps=task["format"] != "txt")

A runtime snapshot is private to the engine. Never serialize or log it. The server
must enforce its usual local configuration authorization before calling update().
No validation operation contacts a service. Tests inject a bytes protector and an
httpx MockTransport; the production default protects keys with current-user DPAPI.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import ipaddress
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import unicodedata
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
import lvats_paths

REMOTE_MODEL_ID = "third-party-api"
DEFAULT_MODEL = "whisper-1"
CHUNK_SECONDS = 600.0
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_CONFIG_BYTES = 64 * 1024
MAX_SEGMENTS = 20000
MAX_CHUNKS = 1000
MAX_TASK_TEXT_BYTES = 1024 * 1024
MAX_TASK_SEGMENTS = 20000
MAX_PROGRESS_CHARS = 20000
TIMESTAMP_TOLERANCE = 0.25  # MP3 encoder padding, not an estimated subtitle track.
CANCEL_POLL_SECONDS = 0.1
REQUEST_TOTAL_SECONDS = 300.0
PROCESS_STOP_SECONDS = 3.0
HTTP_TIMEOUT = httpx.Timeout(connect=20.0, read=180.0, write=180.0, pool=20.0)


class RemoteAPIError(RuntimeError):
    """Only fixed, safe messages cross the engine/server error boundary."""

    def __init__(self, code: str, message: str, http_status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def _error(code: str, message: str, http_status: int | None = None):
    return RemoteAPIError(code, message, http_status)


def _has_controls(value: str) -> bool:
    return any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value)


def _url_text(value, *, allow_empty=False) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise _error("invalid_url", "服务或代理 URL 格式无效。")
    # Check before stripping: CR/LF must not disappear during normalization.
    if _has_controls(value) or "\\" in value:
        raise _error("invalid_url", "服务或代理 URL 包含不允许的字符。")
    value = value.strip()
    if not value and allow_empty:
        return ""
    if not value or any(char.isspace() for char in value) or "?" in value or "#" in value:
        raise _error("invalid_url", "URL 必须有主机，且不能包含查询、片段或空白。")
    return value


def _authority(parts) -> tuple[str, bool]:
    """Return canonical host (IPv6 bracketed) and literal-loopback status."""
    try:
        if not parts.netloc or parts.username is not None or parts.password is not None:
            raise ValueError
        host = parts.hostname
        if not host or "%" in parts.netloc or "@" in parts.netloc:
            raise ValueError
        pattern = (r"\[[^\[\]\s]+\](?::[0-9]{1,5})?" if parts.netloc.startswith("[")
                   else r"[^:\[\]]+(?::[0-9]{1,5})?")
        if not re.fullmatch(pattern, parts.netloc):
            raise ValueError
        # Accessing .port also rejects malformed ports and unmatched brackets.
        port = parts.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        if parts.netloc.endswith(":"):
            raise ValueError
        host = host.rstrip(".").lower()
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            ascii_host = host.encode("idna").decode("ascii").lower()
            if len(ascii_host) > 253 or not all(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in ascii_host.split(".")
            ):
                raise ValueError
            return ascii_host, ascii_host == "localhost"
        mapped = getattr(address, "ipv4_mapped", None)
        loopback = address.is_loopback or (mapped is not None and mapped.is_loopback)
        return (f"[{address.compressed}]" if address.version == 6 else str(address)), loopback
    except (ValueError, UnicodeError):
        raise _error("invalid_url", "URL 主机、端口或认证信息无效。") from None


def normalize_api_url(value: str, *, allow_empty=False) -> str:
    """Accept a base URL or full endpoint; cleartext is loopback-only."""
    value = _url_text(value, allow_empty=allow_empty)
    if not value:
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        raise _error("invalid_url", "服务 URL 格式无效。") from None
    if parts.scheme.lower() not in {"https", "http"}:
        raise _error("invalid_url", "服务 URL 只允许 HTTP 或 HTTPS。")
    host, loopback = _authority(parts)
    if parts.scheme.lower() == "http" and not loopback:
        raise _error("insecure_url", "非回环服务必须使用 HTTPS，避免明文发送密钥。")
    path = parts.path.rstrip("/")
    try:
        if re.search(r"%(?![a-fA-F0-9]{2})", path):
            raise ValueError
        decoded = unquote(path, errors="strict")
        if (_has_controls(decoded) or any(char.isspace() for char in decoded)
                or any(char in decoded for char in "\\?#")
                or any(part in {".", ".."} for part in decoded.split("/"))
                or "//" in decoded):
            raise ValueError
        # Do not normalize escaped separators differently from the HTTP client.
        if "/" in unquote(path.replace("/", ""), errors="strict"):
            raise ValueError
        path = path.encode("ascii").decode("ascii")
    except (ValueError, UnicodeError):
        raise _error("invalid_url", "服务 URL 路径无效。") from None
    if "/audio/transcriptions" in path and not path.endswith("/audio/transcriptions"):
        raise _error("invalid_url", "转录端点不能包含额外的路径。")
    if not path.endswith("/audio/transcriptions"):
        path += "/audio/transcriptions"
    port = parts.port
    authority = host + (f":{port}" if port is not None else "")
    return urlunsplit((parts.scheme.lower(), authority, path, "", ""))


def _port(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise _error("invalid_proxy", "代理端口必须是 1 到 65535 的整数。")
    if isinstance(value, str) and not re.fullmatch(r"[0-9]{1,5}", value.strip()):
        raise _error("invalid_proxy", "代理端口必须是 1 到 65535 的整数。")
    result = int(value)
    if not 1 <= result <= 65535:
        raise _error("invalid_proxy", "代理端口必须是 1 到 65535 的整数。")
    return result


def normalize_proxy(payload: dict, previous: dict | None = None) -> dict:
    """A separate port and unauthenticated HTTP host; no environment proxies."""
    defaults = previous or {"enabled": False, "url": "", "port": 8080}
    if not isinstance(payload, dict) or set(payload) - {"enabled", "url", "port"}:
        raise _error("invalid_proxy", "代理配置格式无效。")
    enabled = payload.get("enabled", defaults["enabled"])
    if not isinstance(enabled, bool):
        raise _error("invalid_proxy", "代理开关必须为布尔值。")
    port = _port(payload.get("port", defaults["port"]))
    try:
        value = _url_text(payload.get("url", defaults["url"]), allow_empty=True)
        if not value:
            if enabled:
                raise _error("invalid_proxy", "启用代理时必须填写 HTTP 代理主机。")
            return {"enabled": enabled, "url": "", "port": port}
        if "://" not in value:
            value = "http://" + value
        parts = urlsplit(value)
        if parts.scheme.lower() != "http" or parts.path or parts.query or parts.fragment:
            raise ValueError
        host, _ = _authority(parts)
        if parts.port is not None:
            raise ValueError
        return {"enabled": enabled, "url": "http://" + host, "port": port}
    except (ValueError, RemoteAPIError):
        raise _error("invalid_proxy", "代理只允许 HTTP 主机；端口单独填写，不支持认证或路径。") from None


def _proxy_url(proxy: dict) -> str | None:
    return f"{proxy['url']}:{proxy['port']}" if proxy["enabled"] else None


def _validate_route(url: str, proxy: dict):
    # HTTP is loopback-only along the entire route, not just the endpoint name.
    if url.startswith("http://") and proxy["enabled"]:
        _, loopback = _authority(urlsplit(proxy["url"]))
        if not loopback:
            raise _error("insecure_proxy", "回环 HTTP 服务仅允许回环代理，避免通过远端代理明文发送密钥。")


def _model(value) -> str:
    if not isinstance(value, str) or _has_controls(value):
        raise _error("invalid_model", "API 模型名称格式无效。")
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", value):
        raise _error("invalid_model", "API 模型名称需为非空的兼容标识符。")
    return value


def _key(value, *, allow_empty=False) -> str:
    if not isinstance(value, str) or len(value) > 8192 or _has_controls(value):
        raise _error("invalid_api_key", "API Key 格式无效。")
    value = value.strip()
    if not value and allow_empty:
        return ""
    if not value or any(not 33 <= ord(char) <= 126 for char in value):
        raise _error("invalid_api_key", "API Key 必须为不含空白或控制字符的有效令牌。")
    return value


class DPAPIProtector:
    """Current-user Windows DPAPI; no machine-wide flag and no UI prompts."""

    def _crypt(self, value: bytes, *, decrypt: bool) -> bytes:
        if os.name != "nt":
            raise _error("secret_storage_unavailable", "当前平台不支持 Windows 用户密钥加密，配置不可用。")
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        try:
            crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            protect = crypt32.CryptProtectData
            protect.argtypes = [ctypes.POINTER(DATA_BLOB), wintypes.LPCWSTR,
                                ctypes.POINTER(DATA_BLOB), ctypes.c_void_p,
                                ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DATA_BLOB)]
            protect.restype = wintypes.BOOL
            unprotect = crypt32.CryptUnprotectData
            unprotect.argtypes = [ctypes.POINTER(DATA_BLOB), ctypes.c_void_p,
                                  ctypes.POINTER(DATA_BLOB), ctypes.c_void_p,
                                  ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DATA_BLOB)]
            unprotect.restype = wintypes.BOOL
            kernel32.LocalFree.argtypes = [ctypes.c_void_p]
            kernel32.LocalFree.restype = ctypes.c_void_p
            buffer = ctypes.create_string_buffer(value, len(value))
            source = DATA_BLOB(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
            output = DATA_BLOB()
            try:
                if decrypt:
                    success = unprotect(ctypes.byref(source), None, None, None, None, 1,
                                        ctypes.byref(output))
                else:
                    success = protect(ctypes.byref(source), "Lvats remote API key", None,
                                      None, None, 1, ctypes.byref(output))
                # CRYPTPROTECT_UI_FORBIDDEN = 1; deliberately omit LOCAL_MACHINE.
                if not success:
                    raise ValueError
                return ctypes.string_at(output.pbData, output.cbData)
            finally:
                if output.pbData:
                    if decrypt:
                        ctypes.memset(output.pbData, 0, output.cbData)
                    kernel32.LocalFree(ctypes.cast(output.pbData, ctypes.c_void_p))
                ctypes.memset(ctypes.addressof(buffer), 0, len(value))
        except Exception:
            raise _error("secret_storage_error", "当前用户密钥加密或解密失败，请重新配置密钥。") from None

    def protect(self, value: bytes) -> bytes:
        return self._crypt(value, decrypt=False)

    def unprotect(self, value: bytes) -> bytes:
        return self._crypt(value, decrypt=True)


class ConfigStore:
    """Atomic, thread-safe, single-process configuration and private snapshots.

    protector is a test-only injectable object: protect(bytes)/unprotect(bytes).
    API keys are never part of public(), update() results, or stored JSON plaintext.
    Omitted/empty api_key preserves the key; clear_api_key explicitly removes it.
    A no-op apply preserves revision; every actual applied change creates a UUID.
    """

    def __init__(self, path: Path, *, protector=None):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._protector = protector if protector is not None else DPAPIProtector()
        self._platform_error = ("当前平台不支持 Windows 用户密钥加密，配置不可用。"
                                if protector is None and os.name != "nt" else None)
        self._load_error: str | None = None
        self._url = ""
        self._model = DEFAULT_MODEL
        self._proxy = {"enabled": False, "url": "", "port": 8080}
        self._revision = uuid.uuid4().hex
        self._encrypted = ""
        self._load()

    def _unprotect(self, encrypted: str) -> str:
        try:
            raw = base64.b64decode(encrypted, validate=True)
            if not raw or len(raw) > MAX_CONFIG_BYTES:
                raise ValueError
            return _key(self._protector.unprotect(raw).decode("utf-8"))
        except Exception:
            raise _error("secret_storage_error", "保存的密钥无法由当前用户解密，请重新配置或明确清除。") from None

    def _protect(self, key: str) -> str:
        try:
            result = self._protector.protect(key.encode("utf-8"))
            if not isinstance(result, bytes) or not result or len(result) > 32768:
                raise ValueError
            return base64.b64encode(result).decode("ascii")
        except Exception:
            raise _error("secret_storage_error", "API Key 加密失败，配置未应用。") from None

    def _load(self):
        try:
            with self.path.open("rb") as stream:
                content = stream.read(MAX_CONFIG_BYTES + 1)
            if len(content) > MAX_CONFIG_BYTES:
                raise ValueError
            data = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
            if (not isinstance(data, dict) or type(data.get("version")) is not int
                    or data.get("version") != 1
                    or set(data) - {"version", "url", "model", "proxy", "revision",
                                    "api_key_encrypted", "key_protection"}
                    or data.get("key_protection") != "dpapi-current-user-v1"):
                raise ValueError
            url = normalize_api_url(data["url"], allow_empty=True)
            model = _model(data["model"])
            proxy = normalize_proxy(data["proxy"])
            _validate_route(url, proxy)
            revision = data["revision"]
            encrypted = data["api_key_encrypted"]
            if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{32}", revision)
                    or not isinstance(encrypted, str) or len(encrypted) > 45000):
                raise ValueError
            self._url, self._model, self._proxy = url, model, proxy
            self._revision, self._encrypted = revision, encrypted
            if encrypted and not self._platform_error:
                self._unprotect(encrypted)
        except FileNotFoundError:
            return
        except Exception:
            # Never expose parser/decryption details, paths, or configuration text.
            self._load_error = "第三方 API 配置读取失败，请重新输入密钥或明确清除后应用。"

    def public(self) -> dict:
        with self._lock:
            error = self._platform_error or self._load_error
            result = {"url": self._url, "model": self._model, "proxy": dict(self._proxy),
                      "api_key_set": bool(self._encrypted),
                      "ready": bool(self._url and self._encrypted and not error),
                      "revision": self._revision}
            if error:
                result["error"] = error
            return result

    def _persist(self, data: dict):
        temporary: Path | None = None
        try:
            encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            if len(encoded) > MAX_CONFIG_BYTES:
                raise ValueError
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".remote-api-config-", suffix=".tmp",
                                               dir=str(self.path.parent))
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception:
            raise _error("config_save_failed", "第三方 API 配置保存失败，原配置未应用更改。") from None
        finally:
            if temporary is not None and temporary.exists():
                # Only the exact file created by mkstemp, in the specified directory.
                if temporary.resolve().parent == self.path.parent.resolve():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass

    def update(self, payload: dict) -> dict:
        with self._lock:
            if (not isinstance(payload, dict)
                    or set(payload) - {"url", "model", "api_key", "clear_api_key", "proxy"}):
                raise _error("invalid_config", "第三方 API 配置格式无效。")
            if self._platform_error:
                raise _error("secret_storage_unavailable", self._platform_error)
            clear = payload.get("clear_api_key", False)
            if not isinstance(clear, bool):
                raise _error("invalid_api_key", "清除密钥开关必须为布尔值。")
            key = _key(payload.get("api_key", ""), allow_empty=True)
            if clear and key:
                raise _error("invalid_api_key", "不能同时清除并设置 API Key。")
            if self._load_error and not (clear or key):
                raise _error("config_invalid", self._load_error)
            url = normalize_api_url(payload.get("url", self._url), allow_empty=True)
            model = _model(payload.get("model", self._model))
            proxy = normalize_proxy(payload.get("proxy", {}), self._proxy)
            _validate_route(url, proxy)
            encrypted = self._encrypted
            changed_key = False
            if clear:
                changed_key = bool(encrypted)
                encrypted = ""
            elif key:
                previous_key = None
                if encrypted and not self._load_error:
                    previous_key = self._unprotect(encrypted)
                if key != previous_key:
                    encrypted = self._protect(key)
                    changed_key = True
            changed = (bool(self._load_error) or changed_key or url != self._url
                       or model != self._model or proxy != self._proxy)
            if not changed:
                return self.public()
            revision = uuid.uuid4().hex
            self._persist({"version": 1, "url": url, "model": model, "proxy": proxy,
                           "revision": revision, "api_key_encrypted": encrypted,
                           "key_protection": "dpapi-current-user-v1"})
            self._url, self._model, self._proxy = url, model, proxy
            self._revision, self._encrypted = revision, encrypted
            self._load_error = None
            return self.public()

    def runtime(self, expected_revision=None) -> dict:
        with self._lock:
            if expected_revision is not None and expected_revision != self._revision:
                raise _error("config_changed", "第三方 API 配置已改变，请确认后重新创建任务。")
            if not self.public()["ready"]:
                raise _error("not_configured", "第三方 API 尚未配置完成或密钥不可用。")
            return {"url": self._url, "model": self._model, "proxy": dict(self._proxy),
                    "proxy_url": _proxy_url(self._proxy),
                    "api_key": self._unprotect(self._encrypted), "revision": self._revision}


class _Cancelled(Exception):
    pass


def _cancelled(cancel) -> bool:
    return bool(cancel is not None and cancel.is_set())


async def _await_cancel(awaitable, cancel):
    """Cancel only this operation, including its HTTP stream or FFmpeg wait."""
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            if _cancelled(cancel):
                raise _Cancelled
            done, _ = await asyncio.wait({task}, timeout=CANCEL_POLL_SECONDS)
            if done:
                if _cancelled(cancel):
                    raise _Cancelled
                return task.result()
    finally:
        if not task.done():
            task.cancel()
        # Retrieve every exception, including when outer task cancellation wins.
        await asyncio.gather(task, return_exceptions=True)


async def _terminate_owned_process(process):
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=PROCESS_STOP_SECONDS)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


async def _run_process(arguments: list[str], cancel, *, capture=False, timeout=60.0) -> bytes:
    if _cancelled(cancel):
        raise _Cancelled
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
        )
    except (OSError, ValueError, NotImplementedError):
        raise _error("media_tool_unavailable", "无法启动 FFmpeg/ffprobe，请检查本机音频工具及异步运行环境。") from None

    async def complete():
        content = bytearray()
        if capture:
            while True:
                block = await process.stdout.read(4096)
                if not block:
                    break
                if len(content) + len(block) > 65536:
                    raise _error("invalid_media", "媒体时长探测输出超限。")
                content.extend(block)
        code = await process.wait()
        if code != 0:
            raise _error("media_conversion_failed", "媒体时长探测或音频抽取失败。")
        return bytes(content)

    try:
        return await _await_cancel(asyncio.wait_for(complete(), timeout=timeout), cancel)
    except asyncio.TimeoutError:
        raise _error("media_timeout", "本机媒体处理超时，任务已停止。") from None
    finally:
        await asyncio.shield(_terminate_owned_process(process))


async def _probe_duration(source: Path, cancel) -> float:
    data = await _run_process([
        lvats_paths.resolve_bundled_tool("ffprobe"), "-v", "error",
        "-protocol_whitelist", "file,pipe", "-show_entries", "format=duration",
        "-of", "json", str(source),
    ], cancel, capture=True, timeout=60.0)
    try:
        value = json.loads(data.decode("utf-8"))["format"]["duration"]
        if isinstance(value, bool):
            raise ValueError
        duration = float(value)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
        if math.ceil(duration / CHUNK_SECONDS) > MAX_CHUNKS:
            raise _error("media_too_long", "媒体长度超过第三方转录的安全分块上限。")
        return duration
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        raise _error("invalid_media", "无法取得有效的媒体时长。") from None


async def _extract_chunk(source: Path, destination: Path, offset: float,
                         duration: float, cancel):
    await _run_process([
        lvats_paths.resolve_bundled_tool("ffmpeg"), "-nostdin", "-hide_banner",
        "-loglevel", "error", "-y", "-protocol_whitelist", "file,pipe",
        "-ss", f"{offset:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
        "-map_chapters", "-1", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
        "-b:a", "32k", "-write_xing", "0", "-f", "mp3", str(destination),
    ], cancel, timeout=max(90.0, duration * 2.0))


def _http_error(status: int) -> RemoteAPIError:
    if 300 <= status < 400:
        return _error("redirect_blocked", "第三方服务返回重定向，已阻止继续发送认证信息。", status)
    if status in {401, 403}:
        return _error("authentication_failed", "第三方服务认证失败，请检查 API Key 与权限。", status)
    if status == 407:
        return _error("proxy_authentication_required", "HTTP 代理要求认证；当前仅支持无认证代理。", status)
    if status == 413:
        return _error("upload_too_large", "第三方服务拒绝当前音频块大小。", status)
    if status == 429:
        return _error("rate_limited", "第三方服务限流或配额不足；未自动重试。", status)
    if status in {408, 504}:
        return _error("timeout", "第三方服务处理超时；未自动重试。", status)
    if status in {400, 404, 405, 415, 422}:
        return _error("protocol_unsupported", "第三方服务不接受当前音频转录协议或参数。", status)
    if status >= 500:
        return _error("service_error", "第三方服务出现错误；未自动重试。", status)
    return _error("http_error", "第三方服务返回了非成功状态。", status)


def _reject_constant(value):
    raise ValueError


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _parse_response(content: bytes, duration: float, offset: float,
                    need_timestamps: bool) -> tuple[str, list[dict]]:
    if len(content) > MAX_RESPONSE_BYTES:
        raise _error("response_too_large", "第三方响应超过安全大小限制。")
    try:
        data = json.loads(content.decode("utf-8"), parse_constant=_reject_constant,
                          object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError):
        raise _error("invalid_response", "第三方响应不是有效的 JSON 转录结果。") from None
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise _error("invalid_response", "第三方响应缺少有效的转录文本。")
    text = data["text"].strip()
    if not need_timestamps:
        return text, []
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) > MAX_SEGMENTS:
        raise _error("timestamps_unsupported", "第三方服务未提供有效的字幕分段；请使用 TXT 或兼容字幕的服务。")
    segments: list[dict] = []
    previous_end = 0.0
    for item in raw_segments:
        try:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError
            start, end = item["start"], item["end"]
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, (int, float)) or not isinstance(end, (int, float))):
                raise ValueError
            start, end = float(start), float(end)
            if (not math.isfinite(start) or not math.isfinite(end) or start < 0
                    or end < start or start < previous_end
                    or start > duration or end > duration + TIMESTAMP_TOLERANCE):
                raise ValueError
            end = min(end, duration)  # Only tolerate codec padding at a chunk's end.
            # Empty cue lines terminate SRT/VTT blocks; normalize only whitespace.
            segment_text = " ".join(item["text"].split())
            if segment_text and end <= start:
                raise ValueError
            previous_end = end
            if segment_text:
                global_start = round(offset + start, 3)
                global_end = round(offset + end, 3)
                if global_end <= global_start:
                    raise ValueError
                segments.append({"start": global_start, "end": global_end,
                                 "text": segment_text})
        except (ValueError, KeyError, OverflowError):
            raise _error("invalid_timestamps", "第三方字幕时间轴无效、无序或超出音频范围。") from None
    # Never publish an SRT/VTT that silently omits transcript content.
    compact = lambda value: "".join(value.split())
    if compact(text) != compact("".join(item["text"] for item in segments)):
        raise _error("timestamps_unsupported", "第三方字幕分段未覆盖完整文本，已停止生成字幕；请使用 TXT。")
    if not segments:
        raise _error("timestamps_unsupported", "第三方服务未提供有效的字幕分段；请使用 TXT。")
    return text, segments


async def _send_chunk(client: httpx.AsyncClient, path: Path, config: dict,
                      prompt: str | None, duration: float, offset: float,
                      need_timestamps: bool) -> tuple[str, list[dict]]:
    data = {"model": config["model"],
            "response_format": "verbose_json" if need_timestamps else "json"}
    if need_timestamps:
        data["timestamp_granularities[]"] = "segment"
    if prompt:
        data["prompt"] = prompt
    headers = {"Authorization": "Bearer " + config["api_key"],
               "Accept": "application/json", "Accept-Encoding": "identity"}
    try:
        with path.open("rb") as audio:
            size = os.fstat(audio.fileno()).st_size
            if size <= 0:
                raise _error("media_conversion_failed", "抽取的音频块为空。")
            if size >= MAX_UPLOAD_BYTES:
                raise _error("upload_too_large", "抽取的音频块超过安全上传限制，未发送。")
            async with client.stream("POST", config["url"], data=data,
                                     files={"file": ("audio.mp3", audio, "audio/mpeg")},
                                     headers=headers) as response:
                if not 200 <= response.status_code < 300:
                    raise _http_error(response.status_code)
                # Identity avoids decompression bombs and bounds bytes before parsing.
                if response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
                    raise _error("invalid_response", "第三方响应使用了不支持的压缩编码。")
                length = response.headers.get("content-length")
                if length is not None:
                    try:
                        length = int(length)
                        if length < 0:
                            raise ValueError
                    except ValueError:
                        raise _error("invalid_response", "第三方响应大小标记无效。") from None
                    if length > MAX_RESPONSE_BYTES:
                        raise _error("response_too_large", "第三方响应超过安全大小限制。")
                content = bytearray()
                async for block in response.aiter_bytes(chunk_size=65536):
                    if len(content) + len(block) > MAX_RESPONSE_BYTES:
                        raise _error("response_too_large", "第三方响应超过安全大小限制。")
                    content.extend(block)
                text, segments = _parse_response(bytes(content), duration, offset, need_timestamps)
                # A provider may echo credentials in a nominally successful result.
                # Fail before callbacks, SSE, persistence or output files see that content.
                key = config["api_key"]
                if key in text or any(key in segment["text"] for segment in segments):
                    raise _error("unsafe_response", "第三方响应包含认证信息，已阻止显示和保存。")
                return text, segments
    except httpx.TimeoutException:
        raise _error("timeout", "第三方请求超时；未自动重试。") from None
    except httpx.ProxyError:
        raise _error("proxy_unavailable", "HTTP 代理连接失败，请检查代理配置。") from None
    except httpx.ConnectError:
        if config.get("proxy_url"):
            raise _error("proxy_unavailable", "通过 HTTP 代理连接服务失败，请检查代理及目标服务。") from None
        raise _error("network_error", "无法连接第三方服务，请检查网络与服务配置。") from None
    except httpx.RequestError:
        raise _error("network_error", "第三方网络请求失败；未自动重试。") from None
    except httpx.HTTPError:
        raise _error("invalid_response", "第三方 HTTP 响应或数据流无效。") from None
    except (ValueError, UnicodeError):
        raise _error("invalid_request", "第三方请求参数编码无效。") from None
    except OSError:
        raise _error("media_read_failed", "无法读取抽取的音频块。") from None


async def _report(callback, percent: float, message: str, partial=None, meta=None):
    if callback is None:
        return
    try:
        try:
            result = callback(percent, message, partial, meta)
        except TypeError:
            result = callback(percent, message)
        if inspect.isawaitable(result):
            await result
    except Exception:
        # UI callbacks must neither break transcription nor expose their exceptions.
        pass


def _runtime_snapshot(config: dict) -> dict:
    if not isinstance(config, dict):
        raise _error("not_configured", "第三方 API 运行配置无效。")
    url = normalize_api_url(config.get("url", ""))
    model = _model(config.get("model", DEFAULT_MODEL))
    key = _key(config.get("api_key", ""))
    proxy = normalize_proxy(config.get("proxy", {}))
    _validate_route(url, proxy)
    return {"url": url, "model": model, "api_key": key,
            "proxy": proxy, "proxy_url": _proxy_url(proxy)}


async def transcribe(audio_path: Path, config: dict, prompt=None, progress_cb=None,
                     cancel=None, need_timestamps=False, job_id="") -> dict:
    """Upload only locally extracted mono MP3 chunks, never the original media.

    Cancel events return partial completed chunks with cancelled=True; externally
    cancelling this asyncio task propagates CancelledError after owned cleanup.
    Remote cancellation cannot revoke already uploaded audio or incurred charges.
    Any error is final: no retries, redirects, local fallback, or made-up segments.
    """
    texts: list[str] = []
    segments: list[dict] = []
    text_bytes = 0
    workspace = None
    try:
        if _cancelled(cancel):
            raise _Cancelled
        snapshot = _runtime_snapshot(config)
        if prompt is not None and (not isinstance(prompt, str) or len(prompt) > 8000):
            raise _error("invalid_prompt", "提示词格式无效或超过长度限制。")
        try:
            source = Path(audio_path).resolve(strict=True)
            if not source.is_file():
                raise OSError
        except (OSError, ValueError, TypeError):
            raise _error("invalid_media", "请选择有效的本机媒体文件。") from None
        duration = await _probe_duration(source, cancel)
        # Defense in depth also applies when duration providers are replaced in tests.
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise _error("invalid_media", "无法取得有效的媒体时长。")
        total_chunks = math.ceil(duration / CHUNK_SECONDS)
        if total_chunks > MAX_CHUNKS:
            raise _error("media_too_long", "媒体长度超过第三方转录的安全分块上限。")
        try:
            temp_root = Path(lvats_paths.TEMP_DIR).resolve()
            temp_root.mkdir(parents=True, exist_ok=True)
            workspace = tempfile.TemporaryDirectory(prefix="remote-api-", dir=str(temp_root))
            working_path = Path(workspace.name).resolve()
            if working_path.parent != temp_root or not working_path.name.startswith("remote-api-"):
                raise OSError
            chunk_path = working_path / "audio.mp3"
        except OSError:
            raise _error("temp_storage_failed", "无法创建隔离音频临时目录。") from None
        await _report(progress_cb, 0.0, "准备第三方转录（仅发送抽取的音频）", None,
                      {"log": "第三方 API 转录开始；不使用本地模型或自动重试。"})
        try:
            async with httpx.AsyncClient(proxy=snapshot["proxy_url"], trust_env=False,
                                         follow_redirects=False, verify=True,
                                         timeout=HTTP_TIMEOUT) as client:
                for index in range(total_chunks):
                    if _cancelled(cancel):
                        raise _Cancelled
                    offset = index * CHUNK_SECONDS
                    chunk_duration = min(CHUNK_SECONDS, duration - offset)
                    await _extract_chunk(source, chunk_path, offset, chunk_duration, cancel)
                    await _report(progress_cb, (index + 0.1) / total_chunks * 95.0,
                                  f"第三方转录中（{index + 1}/{total_chunks} 块）")
                    if _cancelled(cancel):
                        raise _Cancelled
                    try:
                        text, chunk_segments = await _await_cancel(asyncio.wait_for(
                            _send_chunk(client, chunk_path, snapshot, prompt, chunk_duration,
                                        offset, bool(need_timestamps)),
                            timeout=REQUEST_TOTAL_SECONDS), cancel)
                    except asyncio.TimeoutError:
                        raise _error("timeout", "第三方请求超过总时限；未自动重试。") from None
                    additional_bytes = len(text.encode("utf-8")) + (1 if texts and text else 0)
                    if (text_bytes + additional_bytes > MAX_TASK_TEXT_BYTES
                            or len(segments) + len(chunk_segments) > MAX_TASK_SEGMENTS):
                        raise _error("response_too_large", "第三方转录累计结果超过安全限制，任务已停止。")
                    text_bytes += additional_bytes
                    if text:
                        texts.append(text)
                    segments.extend(chunk_segments)
                    try:
                        # Keep just one small extracted block on disk at a time.
                        if chunk_path.parent.resolve() != working_path:
                            raise OSError
                        chunk_path.unlink()
                    except OSError:
                        raise _error("temp_cleanup_failed", "本次音频块清理失败，任务已停止。") from None
                    await _report(progress_cb, (index + 1) / total_chunks * 95.0,
                                  f"第三方转录已完成 {index + 1}/{total_chunks} 块",
                                  "\n".join(texts)[-MAX_PROGRESS_CHARS:],
                                  {"timing": {"kind": "remote", "chunk": index + 1,
                                              "total": total_chunks}})
        except (httpx.RequestError, ValueError):
            raise _error("network_error", "第三方网络客户端配置或连接失败。") from None
        if _cancelled(cancel):
            raise _Cancelled
        result = {"text": "\n".join(texts), "segments": segments,
                  "cancelled": False, "degraded": False}
        if total_chunks > 1:
            result["warning"] = "音频按最多 10 分钟分块转录，跨块语境可能影响识别。"
        await _report(progress_cb, 100.0, "第三方转录完成", result["text"][-MAX_PROGRESS_CHARS:],
                      {"log": f"第三方 API 转录完成：{total_chunks} 个音频块。"})
        return result
    except _Cancelled:
        result = {"text": "\n".join(texts), "segments": segments,
                  "cancelled": True, "degraded": False,
                  "warning": "已停止本机请求；第三方可能已处理上传的音频或产生费用。"}
        await _report(progress_cb, 100.0, "第三方转录已取消", result["text"][-MAX_PROGRESS_CHARS:],
                      {"timing": {"kind": "cancelled"}})
        return result
    except RemoteAPIError:
        raise
    except Exception:
        # Unexpected tool/transport exceptions must not leak request or media data.
        # CancelledError is a BaseException and intentionally propagates.
        raise _error("remote_internal_error", "第三方转录遇到内部错误，任务已停止且未重试。") from None
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except OSError:
                raise _error("temp_cleanup_failed", "本次音频临时目录清理失败，请检查文件占用。") from None
