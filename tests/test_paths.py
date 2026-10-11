import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ASR_TEST_MODE", "1")
os.environ.setdefault("ASR_PLACEHOLDER", "1")
import lvats_paths


class PathLayoutTests(unittest.TestCase):
    def test_windows_program_layout_uses_local_app_data(self):
        with tempfile.TemporaryDirectory() as app, tempfile.TemporaryDirectory() as local:
            with patch.dict(os.environ, {"LOCALAPPDATA": local}, clear=False):
                layout = lvats_paths.resolve_paths(app_root=app, test_mode=False)
            self.assertFalse(layout.portable_mode)
            self.assertEqual(layout.data_root, Path(local).resolve() / "Lvats")
            self.assertEqual(layout.model_dir, layout.data_dir / "models")

    def test_portable_layout_follows_moved_app(self):
        with tempfile.TemporaryDirectory(prefix="Lvats 中文 path ") as parent:
            first = Path(parent) / "first"
            second = Path(parent) / "moved location"
            first.mkdir()
            (first / "portable.flag").touch()
            layout = lvats_paths.resolve_paths(app_root=first, test_mode=False)
            self.assertEqual(layout.data_root, first / "user-data")
            first.rename(second)
            moved = lvats_paths.resolve_paths(app_root=second, test_mode=False)
            self.assertEqual(moved.data_root, second / "user-data")

    def test_test_mode_is_process_isolated(self):
        with tempfile.TemporaryDirectory() as app:
            layout = lvats_paths.resolve_paths(app_root=app, test_mode=True)
            self.assertIn("Lvats-tests", str(layout.data_root))
            self.assertNotEqual(layout.data_root, Path(app))

    def test_explicit_data_root_supports_read_only_app_root(self):
        with tempfile.TemporaryDirectory() as app, tempfile.TemporaryDirectory() as data:
            layout = lvats_paths.resolve_paths(app_root=app, data_root=data, test_mode=False)
            lvats_paths.ensure_directories(layout)
            self.assertTrue(layout.update_dir.is_dir())
            self.assertFalse((Path(app) / "data").exists())

    def test_cache_and_temp_can_be_recreated(self):
        with tempfile.TemporaryDirectory() as app, tempfile.TemporaryDirectory() as data:
            layout = lvats_paths.resolve_paths(app_root=app, data_root=data, test_mode=False)
            lvats_paths.ensure_directories(layout)
            shutil.rmtree(layout.cache_dir)
            shutil.rmtree(layout.temp_dir)
            lvats_paths.ensure_directories(layout)
            self.assertTrue(layout.cache_dir.is_dir())
            self.assertTrue(layout.temp_dir.is_dir())

    def test_migration_is_idempotent_and_preserves_conflicts(self):
        with tempfile.TemporaryDirectory() as legacy, tempfile.TemporaryDirectory() as data:
            old = Path(legacy)
            (old / "state").mkdir()
            (old / "state" / "queue.json").write_text('{"old": 1}', encoding="utf-8")
            (old / "models" / "sample").mkdir(parents=True)
            (old / "models" / "sample" / "config.json").write_text("old", encoding="utf-8")
            layout = lvats_paths.resolve_paths(app_root=old, data_root=data, test_mode=False)

            preview = lvats_paths.migration_preview(old, layout)
            self.assertEqual(preview["counts"]["copy"], 2)
            first = lvats_paths.migrate_legacy_data(old, layout)
            self.assertTrue(first["apply"])
            second = lvats_paths.migrate_legacy_data(old, layout)
            self.assertTrue(all(item["action"] == "same" for item in second["results"]))

            target = layout.state_dir / "queue.json"
            target.write_text('{"new": 2}', encoding="utf-8")
            conflict = lvats_paths.migrate_legacy_data(old, layout)
            queue = next(item for item in conflict["results"] if item["source"].endswith("queue.json"))
            self.assertEqual(queue["action"], "conflict")
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"new": 2})
            self.assertEqual(json.loads(Path(queue["copied_to"]).read_text(encoding="utf-8")), {"old": 1})

    def test_migration_detects_two_legacy_sources_for_same_target(self):
        with tempfile.TemporaryDirectory() as legacy, tempfile.TemporaryDirectory() as data:
            old = Path(legacy)
            first = old / "models" / "Qwen3-ASR-1.7B-hf"
            second = old / "Qwen3-ASR-1.7B-hf"
            first.mkdir(parents=True)
            second.mkdir()
            (first / "config.json").write_text("models copy", encoding="utf-8")
            (second / "config.json").write_text("root copy", encoding="utf-8")
            layout = lvats_paths.resolve_paths(app_root=old, data_root=data, test_mode=False)
            preview = lvats_paths.migration_preview(old, layout)
            self.assertEqual(preview["counts"]["conflict"], 1)
            result = lvats_paths.migrate_legacy_data(old, layout)
            configs = [item for item in result["results"] if item["source"].endswith("config.json")]
            self.assertEqual({item["action"] for item in configs}, {"copy", "conflict"})
            self.assertEqual(len({item["copied_to"] for item in configs}), 2)

    def test_log_rotation_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "lvats.log"
            log.write_bytes(b"12345")
            self.assertTrue(lvats_paths.rotate_log_file(log, max_bytes=4, backups=2))
            self.assertTrue(Path(str(log) + ".1").is_file())
            log.write_bytes(b"67890")
            lvats_paths.rotate_log_file(log, max_bytes=4, backups=2)
            self.assertTrue(Path(str(log) + ".2").is_file())


if __name__ == "__main__":
    unittest.main()
