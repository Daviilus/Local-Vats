"""Detached Lvats restart helper.

The running service starts this helper before exiting.  The helper waits for the
old process to disappear, then reuses the production PowerShell launcher.  Test
mode deliberately starts server.py directly so isolated validation never touches
the protected production service or requires CUDA.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def write_status(path: Path, token: str, phase: str, message: str, **extra):
    payload = {
        "token": token,
        "phase": phase,
        "message": message,
        "updated_at": time.time(),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def process_exists(pid: int) -> bool:
    if os.name == "nt":
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_old_process(pid: int, timeout: float = 25.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_exists(pid):
            return
        time.sleep(0.2)
    raise RuntimeError(f"旧服务进程 PID {pid} 未在 {int(timeout)} 秒内退出")


def windows_flags() -> int:
    return (getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0))


def start_test_service(root: Path, port: int, command_json: str | None) -> int:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    stdout = (state / f"server_{port}.restart.out.log").open("ab")
    stderr = (state / f"server_{port}.restart.err.log").open("ab")
    try:
        command = json.loads(command_json) if command_json else [sys.executable, str(root / "server.py")]
        if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
            raise RuntimeError("隔离重启命令格式无效")
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=windows_flags(),
        )
        return process.pid
    finally:
        stdout.close()
        stderr.close()


def start_production_service(root: Path, port: int) -> int | None:
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    starter = root / "scripts" / "start_lvats.ps1"
    result = subprocess.run(
        [str(powershell), "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(starter), "-ProjectRoot", str(root), "-Port", str(port)],
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=50,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        reason = (result.stderr or result.stdout or f"退出码 {result.returncode}").strip()
        raise RuntimeError(f"Windows 启动脚本失败：{reason[-1200:]}")
    return None


def fetch_health(url: str) -> dict | None:
    context = ssl._create_unverified_context() if url.startswith("https:") else None
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=2, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def wait_for_new_service(url: str, old_pid: int, expected_version: str, timeout: float = 50.0) -> dict:
    deadline = time.monotonic() + timeout
    last_health = None
    while time.monotonic() < deadline:
        health = fetch_health(url)
        if health:
            last_health = health
            if health.get("pid") != old_pid and health.get("version") == expected_version:
                return health
        time.sleep(0.5)
    if last_health:
        raise RuntimeError(
            f"新服务版本或进程不符合预期：期望 v{expected_version}，实际 v{last_health.get('version')}，"
            f"PID {last_health.get('pid')}"
        )
    raise RuntimeError(f"新服务未在 {int(timeout)} 秒内通过健康检查")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--old-pid", required=True, type=int)
    parser.add_argument("--token", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--scheme", choices=("http", "https"), required=True)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-command-json")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    status_file = Path(args.status_file).resolve()
    url = f"{args.scheme}://127.0.0.1:{args.port}"
    try:
        write_status(status_file, args.token, "stopping", "队列已保存，正在等待旧服务退出", old_pid=args.old_pid)
        wait_for_old_process(args.old_pid)
        write_status(status_file, args.token, "starting", "旧服务已退出，正在启动新服务", old_pid=args.old_pid)
        new_pid = (start_test_service(root, args.port, args.test_command_json)
                   if args.test_mode else start_production_service(root, args.port))
        health = wait_for_new_service(url, args.old_pid, args.expected_version)
        write_status(
            status_file, args.token, "ready", f"Lvats v{health['version']} 已恢复服务",
            old_pid=args.old_pid, new_pid=health.get("pid") or new_pid, version=health.get("version"),
        )
        return 0
    except Exception as exc:
        write_status(status_file, args.token, "failed", str(exc), old_pid=args.old_pid)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
