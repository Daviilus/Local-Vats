# -*- coding: utf-8 -*-
"""Centralized Lvats application/data paths and safe legacy migration helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def _default_app_root() -> Path:
    override = os.environ.get("ASR_APP_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _default_resource_root(app_root: Path) -> Path:
    override = os.environ.get("ASR_RESOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    bundle = getattr(sys, "_MEIPASS", None)
    return Path(bundle).resolve() if bundle else app_root


def _installed_data_root() -> Path:
    override = os.environ.get("LOCALAPPDATA")
    if override:
        return Path(override).expanduser().resolve() / "Lvats"
    if os.name == "nt":
        raise RuntimeError("Windows Program 模式无法确定 LOCALAPPDATA")
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share") / "Lvats"


@dataclass(frozen=True)
class PathLayout:
    app_root: Path
    resource_root: Path
    runtime_root: Path
    portable_mode: bool
    test_mode: bool
    data_root: Path
    config_dir: Path
    log_dir: Path
    cache_dir: Path
    data_dir: Path
    temp_dir: Path
    update_dir: Path
    model_dir: Path
    output_dir: Path
    state_dir: Path
    ssl_dir: Path
    runtime_dir: Path
    upload_dir: Path
    wav_cache_dir: Path

    def public_dict(self) -> dict[str, str | bool]:
        result: dict[str, str | bool] = {}
        for key, value in asdict(self).items():
            result[key.upper()] = str(value) if isinstance(value, Path) else value
        return result


def resolve_paths(
    *,
    app_root: str | Path | None = None,
    resource_root: str | Path | None = None,
    data_root: str | Path | None = None,
    portable_mode: bool | None = None,
    test_mode: bool | None = None,
) -> PathLayout:
    """Resolve the complete layout once; callers should not re-detect modes."""
    app = Path(app_root).expanduser().resolve() if app_root else _default_app_root()
    resources = (Path(resource_root).expanduser().resolve() if resource_root
                 else _default_resource_root(app))
    portable = (app / "portable.flag").is_file() if portable_mode is None else portable_mode
    testing = _enabled(os.environ.get("ASR_TEST_MODE")) if test_mode is None else test_mode

    explicit_data = data_root or os.environ.get("ASR_DATA_ROOT")
    if explicit_data:
        root = Path(explicit_data).expanduser().resolve()
    elif testing:
        # A process-scoped default prevents tests from touching real LocalAppData.
        root = Path(tempfile.gettempdir()).resolve() / "Lvats-tests" / str(os.getpid())
    elif portable:
        root = app / "user-data"
    else:
        root = _installed_data_root()

    config = root / "config"
    logs = root / "logs"
    cache = root / "cache"
    data = root / "data"
    temp = root / "temp"
    updates = root / "updates"
    suffix = ".test" if testing else ""
    return PathLayout(
        app_root=app,
        resource_root=resources,
        runtime_root=resources / "runtime",
        portable_mode=portable,
        test_mode=testing,
        data_root=root,
        config_dir=config,
        log_dir=logs,
        cache_dir=cache,
        data_dir=data,
        temp_dir=temp,
        update_dir=updates,
        model_dir=data / "models",
        output_dir=data / f"output{suffix}",
        state_dir=data / "state",
        ssl_dir=config / "ssl",
        runtime_dir=temp / "runtime",
        upload_dir=temp / f"dropcache{suffix}",
        wav_cache_dir=cache / f"wav{suffix}",
    )


PATHS = resolve_paths()
APP_ROOT = PATHS.app_root
RESOURCE_ROOT = PATHS.resource_root
RUNTIME_ROOT = PATHS.runtime_root
PORTABLE_MODE = PATHS.portable_mode
TEST_MODE = PATHS.test_mode
DATA_ROOT = PATHS.data_root
CONFIG_DIR = PATHS.config_dir
LOG_DIR = PATHS.log_dir
CACHE_DIR = PATHS.cache_dir
DATA_DIR = PATHS.data_dir
TEMP_DIR = PATHS.temp_dir
UPDATE_DIR = PATHS.update_dir
MODEL_DIR = PATHS.model_dir
OUTPUT_DIR = PATHS.output_dir
STATE_DIR = PATHS.state_dir
SSL_DIR = PATHS.ssl_dir
RUNTIME_DIR = PATHS.runtime_dir
UPLOAD_DIR = PATHS.upload_dir
WAV_CACHE_DIR = PATHS.wav_cache_dir


def ensure_directories(layout: PathLayout = PATHS) -> None:
    """Create the six standard roots plus required derived directories."""
    for directory in (
        layout.config_dir, layout.log_dir, layout.cache_dir, layout.data_dir,
        layout.temp_dir, layout.update_dir, layout.model_dir, layout.output_dir,
        layout.state_dir, layout.ssl_dir, layout.runtime_dir, layout.upload_dir,
        layout.wav_cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def rotate_log_file(path: str | Path, *, max_bytes: int = 10 * 1024 * 1024,
                    backups: int = 5) -> bool:
    """Rotate ``path`` to .1 ... .N before launch; never touches other files."""
    target = Path(path)
    if not target.is_file() or target.stat().st_size < max_bytes:
        return False
    if backups < 1:
        raise ValueError("backups must be at least 1")
    oldest = target.with_name(target.name + f".{backups}")
    if oldest.exists():
        oldest.unlink()
    for number in range(backups - 1, 0, -1):
        current = target.with_name(target.name + f".{number}")
        if current.exists():
            current.replace(target.with_name(target.name + f".{number + 1}"))
    target.replace(target.with_name(target.name + ".1"))
    return True


def resolve_bundled_tool(name: str, layout: PathLayout = PATHS) -> str:
    """Prefer a bundled executable, then use PATH for source development."""
    executable = name + (".exe" if os.name == "nt" and not name.lower().endswith(".exe") else "")
    for directory in (layout.app_root / "bin", layout.resource_root / "bin"):
        candidate = directory / executable
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name) or name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_pairs(legacy_root: Path, layout: PathLayout):
    pairs = [
        (legacy_root / "config" / "quick_prompts.json", layout.config_dir / "quick_prompts.json"),
        (legacy_root / "config" / "ssl", layout.ssl_dir),
        (legacy_root / "state" / "queue.json", layout.state_dir / "queue.json"),
        (legacy_root / "state" / "notified.json", layout.state_dir / "notified.json"),
        (legacy_root / "state" / "registry.json", layout.cache_dir / "registry.json"),
        (legacy_root / "output", layout.output_dir),
        (legacy_root / ".dropcache", layout.upload_dir),
        (legacy_root / "models", layout.model_dir),
    ]
    for log in sorted((legacy_root / "state").glob("server_*.log")) if (legacy_root / "state").is_dir() else ():
        pairs.append((log, layout.log_dir / log.name))
    for name in (
        "Qwen3-ASR-0.6B-hf", "Qwen3-ASR-1.7B-hf",
        "qwen-3-asr-1.7B-Hf", "qwen3-asr-1.7b-hf",
        "Qwen3-ForcedAligner-0.6B-hf", "qwen3-forcedaligner-0.6b-hf",
    ):
        pairs.append((legacy_root / name, layout.model_dir / name))
    seen_sources: set[str] = set()
    unique_pairs = []
    for source, target in pairs:
        # 大小写别名（如 Qwen3-ASR-1.7B-hf / qwen3-asr-1.7b-hf）在大小写不敏感
        # 文件系统上指向同一真实目录；按解析后的实际路径去重，避免重复计数。
        # 解析结果保留磁盘真实大小写，仅比较时归一化，大小写敏感系统不受影响。
        key = os.path.normcase(str(source.resolve())) if source.exists() else None
        if key is not None:
            if key in seen_sources:
                continue
            seen_sources.add(key)
        unique_pairs.append((source, target))
    return unique_pairs


def _file_action(source: Path, target: Path) -> str:
    if not target.exists():
        return "copy"
    if target.is_file() and source.stat().st_size == target.stat().st_size and _sha256(source) == _sha256(target):
        return "same"
    return "conflict"


def migration_preview(legacy_root: str | Path, layout: PathLayout = PATHS) -> dict:
    source_root = Path(legacy_root).expanduser().resolve()
    entries = []
    planned_targets: dict[Path, Path] = {}
    for source, target in _legacy_pairs(source_root, layout):
        if not source.exists():
            continue
        files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
        for item in files:
            destination = target if source.is_file() else target / item.relative_to(source)
            previous = planned_targets.get(destination)
            if previous is not None:
                action = ("same" if item.stat().st_size == previous.stat().st_size
                          and _sha256(item) == _sha256(previous) else "conflict")
            else:
                action = _file_action(item, destination)
                planned_targets[destination] = item
            entries.append({
                "source": str(item), "target": str(destination),
                "bytes": item.stat().st_size, "action": action,
            })
    counts = {name: sum(1 for entry in entries if entry["action"] == name)
              for name in ("copy", "same", "conflict")}
    return {"legacy_root": str(source_root), "data_root": str(layout.data_root),
            "apply": False, "counts": counts, "entries": entries}


def _conflict_target(target: Path, source_hash: str) -> Path:
    candidate = target.with_name(f"{target.name}.legacy-{source_hash[:12]}")
    number = 1
    while candidate.exists() and _sha256(candidate) != source_hash:
        candidate = target.with_name(f"{target.name}.legacy-{source_hash[:12]}-{number}")
        number += 1
    return candidate


def migrate_legacy_data(legacy_root: str | Path, layout: PathLayout = PATHS) -> dict:
    """Copy and verify legacy data. Existing differing files are preserved on both sides."""
    ensure_directories(layout)
    report = migration_preview(legacy_root, layout)
    results = []
    for entry in report["entries"]:
        source = Path(entry["source"])
        target = Path(entry["target"])
        # Re-evaluate after every copy so two legacy roots targeting the same
        # destination can never overwrite one another based on a stale preview.
        action = _file_action(source, target)
        entry = {**entry, "action": action}
        final_target = target
        source_hash = _sha256(source)
        if action == "copy":
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif action == "conflict":
            final_target = _conflict_target(target, source_hash)
            if not final_target.exists():
                final_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, final_target)
        if _sha256(final_target) != source_hash or final_target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"迁移校验失败：{source} -> {final_target}")
        results.append({**entry, "copied_to": str(final_target), "sha256": source_hash})

    report.update({"apply": True, "completed_at": time.time(), "results": results})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = layout.log_dir / f"migration-{stamp}.json"
    number = 1
    while log_path.exists():
        log_path = layout.log_dir / f"migration-{stamp}-{number}.json"
        number += 1
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["log_file"] = str(log_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lvats 路径检查与旧数据安全迁移")
    parser.add_argument("--json", action="store_true", help="输出当前路径 JSON")
    parser.add_argument("--legacy-root", help="明确指定旧源码版根目录")
    parser.add_argument("--apply", action="store_true", help="复制并校验；默认只预览")
    args = parser.parse_args(argv)
    if args.legacy_root:
        result = (migrate_legacy_data(args.legacy_root) if args.apply
                  else migration_preview(args.legacy_root))
    else:
        result = PATHS.public_dict()
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
