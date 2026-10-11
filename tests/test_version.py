"""Verify the Web source version without requiring desktop build tooling."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
os.environ.setdefault('ASR_TEST_MODE', '1')
os.environ.setdefault('ASR_PLACEHOLDER', '1')
os.environ.setdefault('ASR_PORT', '8001')

import server


def source_version() -> str:
    return (BASE / 'VERSION').read_text(encoding='utf-8').strip()


class WebVersionSourceTests(unittest.TestCase):
    def test_root_version_format(self):
        self.assertRegex(source_version(), r'^\d+\.\d+\.\d+$')

    def test_server_uses_version_file(self):
        self.assertEqual(server.VERSION, source_version())

    def test_no_hardcoded_version_literal_in_server(self):
        text = (BASE / 'server.py').read_text(encoding='utf-8')
        self.assertIsNone(re.search(r'^VERSION\s*=\s*[\"\'][\d.]+[\"\']', text, re.M))

    def test_frontend_version_is_runtime_derived(self):
        text = (BASE / 'static/index.html').read_text(encoding='utf-8')
        self.assertIn('id="verChip"', text)
        self.assertIn('id="footVer"', text)
        self.assertIn('h.version', text)
        self.assertNotIn('>v' + source_version() + '<', text)

    def test_readme_current_version_matches_source(self):
        text = (BASE / 'README.md').read_text(encoding='utf-8')
        self.assertIn('当前 Web 版本为 **' + source_version() + '**', text)

    def test_source_loader_reads_adjacent_version_without_build_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'VERSION').write_text('9.8.7\n', encoding='utf-8')
            with patch.dict(sys.modules, {'lvats_version': None}), \
                 patch.object(server, '__file__', str(root / 'server.py')):
                self.assertEqual(server._load_version(), '9.8.7')


if __name__ == '__main__':
    unittest.main()
