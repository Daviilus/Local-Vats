"""Offline remote API tests: no real keys, models, media, or external requests.

Run with ASR_TEST_MODE=1 ASR_PLACEHOLDER=1 ASR_PORT=8001 and a unique
ASR_DATA_ROOT inside this workspace. All audio bytes are synthetic fixtures;
FFmpeg/ffprobe are mocked. Only routing tests bind ephemeral loopback ports.
"""

import asyncio
import base64
import ctypes
import json
import os
import sys
import tempfile
import threading
import traceback
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ASR_TEST_MODE"] = "1"
os.environ["ASR_PLACEHOLDER"] = "1"
os.environ["ASR_PORT"] = "8001"
os.environ.setdefault("ASR_DATA_ROOT", str(ROOT / ".cowork-temp" / ("remote-unit-" + uuid.uuid4().hex)))
TEST_ROOT = Path(os.environ["ASR_DATA_ROOT"]).resolve()
if ROOT not in TEST_ROOT.parents:
    raise RuntimeError("Remote API tests require an isolated ASR_DATA_ROOT inside the workspace")
TEST_ROOT.mkdir(parents=True, exist_ok=True)

import httpx
import remote_api as remote

REAL_CLIENT = httpx.AsyncClient
FAKE_KEY = "unit-test-token-NOT-A-CREDENTIAL"
SECRET_URL = "https://private-endpoint.example.test/v1"
ORIGINAL_MEDIA = b"SYNTHETIC-PRIVATE-VIDEO-NOT-FOR-UPLOAD"
EXTRACTED_AUDIO = b"SYNTHETIC-MONO-MP3-FIXTURE"


class FakeProtector:
    """Reversible fake, solely to verify the production encryption boundary."""

    def __init__(self):
        self.protect_calls = 0
        self.unprotect_calls = 0

    def protect(self, value):
        self.protect_calls += 1
        return b"test-cipher:" + bytes(byte ^ 0xA7 for byte in value)

    def unprotect(self, value):
        self.unprotect_calls += 1
        if not value.startswith(b"test-cipher:"):
            raise ValueError("Fake decryption rejected")
        return bytes(byte ^ 0xA7 for byte in value[len(b"test-cipher:"):])


class SafeAssertions:
    def assert_safe(self, error, *sensitive):
        rendered = str(error) + repr(error) + "".join(traceback.format_exception(error))
        for item in (FAKE_KEY, SECRET_URL, *sensitive):
            self.assertNotIn(str(item), rendered)


class ConfigTests(SafeAssertions, unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="config-", dir=TEST_ROOT)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config" / "remote.json"
        self.protector = FakeProtector()
        self.store = remote.ConfigStore(self.path, protector=self.protector)

    def configure(self, **extra):
        payload = {"url": SECRET_URL, "api_key": FAKE_KEY}
        payload.update(extra)
        return self.store.update(payload)

    def test_default_public_has_no_secret_and_not_ready(self):
        value = self.store.public()
        self.assertEqual(set(value), {"url", "model", "proxy", "api_key_set", "ready", "revision"})
        self.assertEqual(value["model"], "whisper-1")
        self.assertEqual(value["proxy"], {"enabled": False, "url": "", "port": 8080})
        self.assertFalse(value["ready"])
        self.assertFalse(value["api_key_set"])
        with self.assertRaises(remote.RemoteAPIError) as caught:
            self.store.runtime()
        self.assertEqual(caught.exception.code, "not_configured")
        self.assert_safe(caught.exception)

    def test_encrypted_persistence_reload_and_no_public_key(self):
        initial = self.store.public()["revision"]
        applied = self.configure()
        self.assertTrue(applied["ready"])
        self.assertTrue(applied["api_key_set"])
        self.assertNotEqual(initial, applied["revision"])
        self.assertNotIn(FAKE_KEY, json.dumps(applied))
        disk = self.path.read_text(encoding="utf-8")
        self.assertNotIn(FAKE_KEY, disk)
        data = json.loads(disk)
        self.assertNotIn("api_key", data)
        self.assertEqual(data["key_protection"], "dpapi-current-user-v1")
        self.assertEqual(self.protector.protect_calls, 1)
        self.assertNotEqual(base64.b64decode(data["api_key_encrypted"]), FAKE_KEY.encode())
        reloaded = remote.ConfigStore(self.path, protector=FakeProtector())
        self.assertEqual(reloaded.public(), applied)
        self.assertEqual(reloaded.runtime(applied["revision"])["api_key"], FAKE_KEY)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_omitted_empty_and_same_key_preserve_revision(self):
        public = self.configure()
        self.assertEqual(self.store.update({}), public)
        self.assertEqual(self.store.update({"api_key": ""}), public)
        self.assertEqual(self.store.update({"api_key": "  "}), public)
        self.assertEqual(self.store.update({"api_key": FAKE_KEY}), public)
        self.assertEqual(self.store.runtime()["api_key"], FAKE_KEY)
        self.assertEqual(self.protector.protect_calls, 1)

    def test_clear_key_and_refuse_conflicting_action(self):
        self.configure()
        with self.assertRaises(remote.RemoteAPIError):
            self.store.update({"clear_api_key": True, "api_key": FAKE_KEY})
        before = self.store.public()["revision"]
        cleared = self.store.update({"clear_api_key": True})
        self.assertFalse(cleared["api_key_set"])
        self.assertFalse(cleared["ready"])
        self.assertNotEqual(before, cleared["revision"])
        self.assertEqual(json.loads(self.path.read_text())["api_key_encrypted"], "")
        reloaded = remote.ConfigStore(self.path, protector=FakeProtector())
        self.assertFalse(reloaded.public()["api_key_set"])
        self.assertEqual(self.store.update({"clear_api_key": True}), cleared)

    def test_revision_change_stale_task_failure_and_snapshot_isolation(self):
        previous = self.configure()
        snapshot = self.store.runtime(expected_revision=previous["revision"])
        applied = self.store.update({"model": "custom-whisper-v2"})
        self.assertNotEqual(previous["revision"], applied["revision"])
        with self.assertRaises(remote.RemoteAPIError) as caught:
            self.store.runtime(expected_revision=previous["revision"])
        self.assertEqual(caught.exception.code, "config_changed")
        self.assert_safe(caught.exception)
        self.assertEqual(snapshot["model"], "whisper-1")
        snapshot["proxy"]["url"] = "http://changed"
        self.assertEqual(self.store.public()["proxy"]["url"], "")
        public = self.store.public()
        public["proxy"]["port"] = 1111
        self.assertEqual(self.store.runtime()["proxy"]["port"], 8080)

    def test_setting_new_key_changes_revision(self):
        public = self.configure()
        value = self.store.update({"api_key": FAKE_KEY + "-rotated"})
        self.assertNotEqual(value["revision"], public["revision"])
        self.assertEqual(self.store.runtime()["api_key"], FAKE_KEY + "-rotated")
        self.assertNotIn(FAKE_KEY, self.path.read_text())

    def test_atomic_save_failure_preserves_memory_disk_and_cleans_tmp(self):
        previous = self.configure()
        content = self.path.read_bytes()
        with patch.object(remote.os, "replace", side_effect=OSError(FAKE_KEY + SECRET_URL)):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                self.store.update({"model": "new-model", "api_key": FAKE_KEY + "-new"})
        self.assertEqual(caught.exception.code, "config_save_failed")
        self.assert_safe(caught.exception)
        self.assertEqual(self.path.read_bytes(), content)
        self.assertEqual(self.store.public(), previous)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_protector_error_is_safe_and_does_not_save(self):
        with patch.object(self.protector, "protect", side_effect=RuntimeError(FAKE_KEY + SECRET_URL)):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                self.configure()
        self.assertEqual(caught.exception.code, "secret_storage_error")
        self.assert_safe(caught.exception)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.store.public()["ready"])

    def test_corrupt_or_foreign_config_is_fail_closed_and_recoverable(self):
        self.configure()
        data = json.loads(self.path.read_text())
        data["api_key_encrypted"] = "!!!bad encrypted key!!!"
        self.path.write_text(json.dumps(data), encoding="utf-8")
        broken = remote.ConfigStore(self.path, protector=FakeProtector())
        self.assertFalse(broken.public()["ready"])
        self.assertIn("error", broken.public())
        with self.assertRaises(remote.RemoteAPIError):
            broken.runtime()
        with self.assertRaises(remote.RemoteAPIError):
            broken.update({})
        recovered = broken.update({"api_key": FAKE_KEY})
        self.assertTrue(recovered["ready"])
        self.assertNotIn("error", recovered)
        data = json.loads(self.path.read_text())
        data["api_key"] = FAKE_KEY
        self.path.write_text(json.dumps(data))
        refused = remote.ConfigStore(self.path, protector=FakeProtector())
        self.assertFalse(refused.public()["ready"])
        self.assert_safe(remote.RemoteAPIError("test", refused.public()["error"]))
        self.assertFalse(refused.update({"clear_api_key": True})["api_key_set"])

    def test_oversized_config_is_safe(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b"x" * (remote.MAX_CONFIG_BYTES + 1))
        loaded = remote.ConfigStore(self.path, protector=FakeProtector())
        self.assertFalse(loaded.public()["ready"])
        self.assertIn("error", loaded.public())

    def test_non_windows_default_fails_closed_without_plaintext_file(self):
        # Patch only remote's os reference, never pathlib's global platform state.
        from types import SimpleNamespace
        with patch.object(remote, "os", SimpleNamespace(name="posix")):
            store = remote.ConfigStore(self.path)
            self.assertFalse(store.public()["ready"])
            self.assertIn("error", store.public())
            with self.assertRaises(remote.RemoteAPIError) as caught:
                store.update({"url": SECRET_URL, "api_key": FAKE_KEY})
            self.assertEqual(caught.exception.code, "secret_storage_unavailable")
            self.assert_safe(caught.exception)
        self.assertFalse(self.path.exists())

    def test_invalid_fields_have_no_side_effect_and_safe_errors(self):
        public = self.configure()
        candidates = [None, [], {"unknown": FAKE_KEY}, {"model": ""},
                      {"model": "new\nmodel"}, {"model": "has spaces"},
                      {"api_key": None}, {"api_key": "line\nkey"},
                      {"api_key": "key with spaces"}, {"api_key": "秘密"},
                      {"clear_api_key": 1}, {"proxy": None}]
        for payload in candidates:
            with self.subTest(payload_type=type(payload).__name__):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    self.store.update(payload)
                self.assert_safe(caught.exception)
                self.assertEqual(self.store.public(), public)

    def test_runtime_decryption_failure_remains_safe(self):
        self.configure()
        with patch.object(self.protector, "unprotect", side_effect=RuntimeError(FAKE_KEY + SECRET_URL)):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                self.store.runtime()
        self.assertEqual(caught.exception.code, "secret_storage_error")
        self.assert_safe(caught.exception)

    def test_duplicate_json_keys_rejected(self):
        self.configure()
        content = self.path.read_text()
        self.path.write_text(content[:-1] + ',"url":"https://changed.example.test/v1"}')
        store = remote.ConfigStore(self.path, protector=FakeProtector())
        self.assertFalse(store.public()["ready"])
        self.assertIn("error", store.public())

    def test_proxy_normalization_snapshot_and_route_safety(self):
        public = self.configure(proxy={"enabled": True, "url": "[::1]", "port": "3128"})
        self.assertEqual(public["proxy"], {"enabled": True, "url": "http://[::1]", "port": 3128})
        self.assertEqual(self.store.runtime()["proxy_url"], "http://[::1]:3128")
        disabled = self.store.update({"proxy": {"enabled": False}})
        self.assertFalse(disabled["proxy"]["enabled"])
        self.assertIsNone(self.store.runtime()["proxy_url"])
        self.assertEqual(disabled["proxy"]["url"], "http://[::1]")
        self.store.update({"url": "http://127.0.0.1/v1"})
        with self.assertRaises(remote.RemoteAPIError) as caught:
            self.store.update({"proxy": {"enabled": True, "url": "proxy.example.test", "port": 8080}})
        self.assertEqual(caught.exception.code, "insecure_proxy")
        self.assert_safe(caught.exception)


class DPAPIBindingTests(SafeAssertions, unittest.TestCase):
    def test_dpapi_current_user_flags_ctypes_binding_and_local_free(self):
        from ctypes import wintypes
        from types import SimpleNamespace

        class Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        buffers = []
        calls = []
        def crypt(source, description, entropy, reserved, prompt, flags, destination, decrypt=False):
            self.assertEqual(flags, 1)  # UI_FORBIDDEN only, never LOCAL_MACHINE.
            self.assertIsNone(entropy)
            self.assertIsNone(reserved)
            self.assertIsNone(prompt)
            blob = ctypes.cast(source, ctypes.POINTER(Blob)).contents
            value = ctypes.string_at(blob.pbData, blob.cbData)
            calls.append((decrypt, value, description))
            result = value[len(b"fake-dpapi:"):] if decrypt else b"fake-dpapi:" + value
            buffer = ctypes.create_string_buffer(result, len(result))
            buffers.append((buffer, result, decrypt))
            output = ctypes.cast(destination, ctypes.POINTER(Blob)).contents
            output.cbData = len(result)
            output.pbData = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
            return True

        protect = unittest.mock.Mock(side_effect=crypt)
        unprotect = unittest.mock.Mock(side_effect=lambda *args: crypt(*args, decrypt=True))
        released = []
        def free(pointer):
            released.append(pointer)
            buffer, expected, decrypt = buffers[len(released) - 1]
            if decrypt:
                self.assertEqual(ctypes.string_at(pointer, len(expected)), b"\0" * len(expected))
            return None
        local_free = unittest.mock.Mock(side_effect=free)
        fake_crypt32 = SimpleNamespace(CryptProtectData=protect, CryptUnprotectData=unprotect)
        fake_kernel32 = SimpleNamespace(LocalFree=local_free)
        with patch.object(remote, "os", SimpleNamespace(name="nt")), \
                patch.object(ctypes, "WinDLL", side_effect=[fake_crypt32, fake_kernel32,
                                                           fake_crypt32, fake_kernel32], create=True):
            protector = remote.DPAPIProtector()
            encrypted = protector.protect(FAKE_KEY.encode())
            self.assertEqual(protector.unprotect(encrypted), FAKE_KEY.encode())
        self.assertEqual(calls[0][1], FAKE_KEY.encode())
        self.assertEqual(calls[1][1], encrypted)
        self.assertEqual(calls[0][2], "Lvats remote API key")
        self.assertIsNone(calls[1][2])
        self.assertEqual(local_free.call_count, 2)
        self.assertEqual(protect.restype, wintypes.BOOL)
        self.assertEqual(unprotect.restype, wintypes.BOOL)
        self.assertEqual(local_free.argtypes, [ctypes.c_void_p])

    def test_dpapi_failure_is_safe_and_non_windows_refused(self):
        from types import SimpleNamespace
        protect = unittest.mock.Mock(return_value=0)
        unprotect = unittest.mock.Mock(return_value=0)
        libraries = [SimpleNamespace(CryptProtectData=protect, CryptUnprotectData=unprotect),
                     SimpleNamespace(LocalFree=unittest.mock.Mock())]
        with patch.object(remote, "os", SimpleNamespace(name="nt")), \
                patch.object(ctypes, "WinDLL", side_effect=libraries, create=True):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                remote.DPAPIProtector().protect(FAKE_KEY.encode())
        self.assertEqual(caught.exception.code, "secret_storage_error")
        self.assert_safe(caught.exception)
        with patch.object(remote, "os", SimpleNamespace(name="posix")):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                remote.DPAPIProtector().protect(FAKE_KEY.encode())
        self.assertEqual(caught.exception.code, "secret_storage_unavailable")
        self.assert_safe(caught.exception)


class ValidationTests(SafeAssertions, unittest.TestCase):
    def test_service_url_normalization(self):
        samples = {
            "https://API.example.test/v1/": "https://api.example.test/v1/audio/transcriptions",
            "https://api.example.test/v1/audio/transcriptions/": "https://api.example.test/v1/audio/transcriptions",
            "https://api.example.test": "https://api.example.test/audio/transcriptions",
            "http://127.0.0.1:1234/v1": "http://127.0.0.1:1234/v1/audio/transcriptions",
            "http://LOCALHOST/v1": "http://localhost/v1/audio/transcriptions",
            "http://[::1]:4321/v1": "http://[::1]:4321/v1/audio/transcriptions",
            "http://127.1.2.3/v1": "http://127.1.2.3/v1/audio/transcriptions",
        }
        for value, expected in samples.items():
            with self.subTest(value=value):
                normalized = remote.normalize_api_url(value)
                self.assertEqual(normalized, expected)
                self.assertEqual(remote.normalize_api_url(normalized), normalized)

    def test_service_url_strict_validation_and_no_original_in_error(self):
        invalid = ["", "file:///private", "ftp://host/v1", "javascript:alert(1)",
                   "https:///v1", "api.example.test/v1", "http://api.example.test/v1",
                   "http://192.168.1.1/v1", "http://localhost.evil.test/v1", "http://127.1/v1",
                   "https://user:secret@api.example.test/v1", "https://api.example.test/v1?x=secret",
                   "https://api.example.test/v1?", "https://api.example.test/v1#", "https://host/v1\n",
                   "https://ho\x00st/v1", "https://host/v1/%0a", "https://host/v1/%00",
                   "https://host/v1/%2fsecret", "https://host/v1/%5csecret", "https://host/a/../v1",
                   "https://host/a/%2e%2e/v1", "https://host:0/v1", "https://host:65536/v1",
                   "https://host:/v1", "https://[::1]trailing/v1", "https://[::1/v1",
                   "https://host\\evil/v1", "https://host/a//b", "https://host/v1/%ZZ",
                   "https://host/v1/audio/transcriptions/extra", "https://host/a b", None, 7]
        for value in invalid:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    remote.normalize_api_url(value)
                self.assert_safe(caught.exception)
                if isinstance(value, str) and value:
                    self.assertNotIn(value, str(caught.exception))

    def test_proxy_strict_validation_ports_ipv6_and_host_only(self):
        value = remote.normalize_proxy({"enabled": True, "url": "proxy.example.test", "port": "65535"})
        self.assertEqual(value, {"enabled": True, "url": "http://proxy.example.test", "port": 65535})
        self.assertEqual(remote.normalize_proxy({"url": "http://[::1]", "port": 1})["port"], 1)
        invalid = [{"enabled": True}, {"enabled": "true"}, {"url": "https://proxy"},
                   {"url": "http://proxy/"}, {"url": "http://user:password@proxy"},
                   {"url": "http://proxy/path"}, {"url": "http://proxy?query"},
                   {"url": "proxy#"}, {"url": "http://proxy:3128"}, {"url": "http://[::1]junk"},
                   {"url": "::1"}, {"url": "proxy\n"}, {"port": ""}, {"port": True},
                   {"port": "1.5"}, {"port": 0}, {"port": 65536}, {"port": "-1"},
                   {"port": 2.5}, {"unknown": "x"}, []]
        for payload in invalid:
            with self.subTest(payload_type=type(payload).__name__):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    remote.normalize_proxy(payload)
                self.assertEqual(caught.exception.code, "invalid_proxy")
                self.assert_safe(caught.exception)


class ResponseTests(SafeAssertions, unittest.TestCase):
    @staticmethod
    def response(text="hello", segments=None):
        return json.dumps({"text": text, "segments": segments}).encode()

    def test_txt_does_not_require_segments(self):
        self.assertEqual(remote._parse_response(b'{"text":"hello"}', 10, 0, False), ("hello", []))
        self.assertEqual(remote._parse_response(b'{"text":""}', 10, 0, False), ("", []))

    def test_subtitle_offsets_and_whitespace_coverage(self):
        content = self.response("hello world", [{"start": 0, "end": 1, "text": "hello "},
                                               {"start": 2, "end": 5, "text": " world"}])
        text, segments = remote._parse_response(content, 10, 600, True)
        self.assertEqual(text, "hello world")
        self.assertEqual(segments, [{"start": 600.0, "end": 601.0, "text": "hello"},
                                    {"start": 602.0, "end": 605.0, "text": "world"}])

    def test_missing_empty_or_incomplete_segments_fail_without_fake_track(self):
        samples = [b'{"text":"hello"}', self.response(segments=[]), self.response(segments={}),
                   self.response("", []), self.response("hello world", [{"start": 0, "end": 2, "text": "hello"}])]
        for content in samples:
            with self.subTest(content_size=len(content)):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    remote._parse_response(content, 10, 0, True)
                self.assertEqual(caught.exception.code, "timestamps_unsupported")
                self.assert_safe(caught.exception)

    def test_invalid_json_and_schema_is_safe(self):
        for content in [b"not json " + FAKE_KEY.encode(), b"[]", b'{"text":42}',
                        b'{"text":"a","text":"b"}', b'\xff', b'{"text":"a","x":NaN}']:
            with self.subTest(content_size=len(content)):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    remote._parse_response(content, 10, 0, False)
                self.assertEqual(caught.exception.code, "invalid_response")
                self.assert_safe(caught.exception)

    def test_timestamps_reject_nonfinite_order_bounds_types_or_empty_range(self):
        cases = [(-1, 2), (2, 1), (0, 10.5), (11, 11), (True, 2), (0, "2"),
                 (0, float("inf")), (float("nan"), 2), (0, 0), (0, 0.00001)]
        for start, end in cases:
            with self.subTest(start_type=type(start).__name__, end_type=type(end).__name__):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    remote._parse_response(self.response(segments=[{"start": start, "end": end, "text": "hello"}]),
                                           10, 0, True)
                self.assertIn(caught.exception.code, {"invalid_response", "invalid_timestamps"})
                self.assert_safe(caught.exception)
        for segments in [[{"start": 0, "end": 4, "text": "hel"}, {"start": 3, "end": 5, "text": "lo"}],
                         [{"start": 1, "text": "hello"}], [None]]:
            with self.assertRaises(remote.RemoteAPIError) as caught:
                remote._parse_response(self.response(segments=segments), 10, 0, True)
            self.assertEqual(caught.exception.code, "invalid_timestamps")

    def test_codec_padding_tolerance_is_clamped_and_limited(self):
        content = self.response(segments=[{"start": 0, "end": 10.2, "text": "hello"}])
        self.assertEqual(remote._parse_response(content, 10, 600, True)[1][0]["end"], 610)
        with self.assertRaises(remote.RemoteAPIError):
            remote._parse_response(self.response(segments=[{"start": 0, "end": 11, "text": "hello"}]),
                                   10, 600, True)

    def test_response_and_segment_limits(self):
        with patch.object(remote, "MAX_RESPONSE_BYTES", 3):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                remote._parse_response(b'{}  ', 10, 0, False)
            self.assertEqual(caught.exception.code, "response_too_large")
        with patch.object(remote, "MAX_SEGMENTS", 1):
            with self.assertRaises(remote.RemoteAPIError):
                remote._parse_response(self.response("ab", [{"start": 0, "end": 1, "text": "a"},
                                                            {"start": 1, "end": 2, "text": "b"}]), 10, 0, True)


class AudioProcessTests(SafeAssertions, unittest.IsolatedAsyncioTestCase):
    async def test_probe_duration_valid_invalid_and_no_shell_arguments(self):
        source = TEST_ROOT / "中文 空格 synthetic.mp4"
        with patch.object(remote, "_run_process", AsyncMock(return_value=b'{"format":{"duration":"12.5"}}')) as run:
            self.assertEqual(await remote._probe_duration(source, None), 12.5)
        args = run.call_args.args[0]
        self.assertEqual(args[-1], str(source))
        self.assertIn("-protocol_whitelist", args)
        for data in [b"not JSON", b'{"format":{"duration":"NaN"}}', b'{"format":{"duration":0}}',
                     b'{"format":{"duration":true}}', b'{"format":{"duration":-1}}']:
            with patch.object(remote, "_run_process", AsyncMock(return_value=data)):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    await remote._probe_duration(source, None)
                self.assertEqual(caught.exception.code, "invalid_media")
        with patch.object(remote, "_run_process", AsyncMock(return_value=b'{"format":{"duration":700000}}')):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                await remote._probe_duration(source, None)
            self.assertEqual(caught.exception.code, "media_too_long")

    async def test_extract_command_only_audio_low_bitrate_and_metadata_removed(self):
        source = TEST_ROOT / "中文 with spaces.mp4"
        destination = TEST_ROOT / "owned temp" / "audio.mp3"
        with patch.object(remote, "_run_process", AsyncMock()) as run:
            await remote._extract_chunk(source, destination, 600, 20, None)
        args = run.call_args.args[0]
        for flag in ["-nostdin", "-vn", "-sn", "-dn", "-map_metadata", "-map_chapters"]:
            self.assertIn(flag, args)
        for flag, value in {"-ss": "600.000000", "-t": "20.000000", "-ac": "1",
                            "-ar": "16000", "-b:a": "32k", "-c:a": "libmp3lame"}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(args[args.index("-i") + 1], str(source))
        self.assertEqual(args[-1], str(destination))

    async def test_capture_reads_multiple_pieces_and_subprocess_has_no_shell(self):
        class Reader:
            def __init__(self):
                self.parts = [b'{"format":', b'{"duration":"2"}}', b""]
            async def read(self, size):
                return self.parts.pop(0)

        class Process:
            returncode = 0
            stdout = Reader()
            wait = AsyncMock(return_value=0)
            terminate = unittest.mock.Mock()

        process = Process()
        with patch.object(remote.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)) as spawn:
            content = await remote._run_process(["ffprobe", "中文 path"], None, capture=True)
        self.assertEqual(content, b'{"format":{"duration":"2"}}')
        self.assertNotIn("shell", spawn.call_args.kwargs)
        self.assertEqual(spawn.call_args.args, ("ffprobe", "中文 path"))
        self.assertEqual(spawn.call_args.kwargs["stderr"], remote.subprocess.DEVNULL)
        process.terminate.assert_not_called()

    async def test_cancel_and_timeout_terminate_only_owned_process(self):
        class Process:
            def __init__(self):
                self.returncode = None
                self.ended = asyncio.Event()
                self.terminated = 0
                self.killed = 0
            async def wait(self):
                await self.ended.wait()
                return self.returncode
            def terminate(self):
                self.terminated += 1
                self.returncode = -15
                self.ended.set()
            def kill(self):
                self.killed += 1
                self.returncode = -9
                self.ended.set()

        process = Process()
        cancel = threading.Event()
        with patch.object(remote.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)), \
                patch.object(remote, "CANCEL_POLL_SECONDS", 0.001):
            task = asyncio.create_task(remote._run_process(["ffmpeg"], cancel))
            await asyncio.sleep(0.01)
            cancel.set()
            with self.assertRaises(remote._Cancelled):
                await asyncio.wait_for(task, 1)
        self.assertEqual(process.terminated, 1)
        self.assertEqual(process.killed, 0)
        process = Process()
        with patch.object(remote.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                await remote._run_process(["ffmpeg"], None, timeout=0.01)
        self.assertEqual(caught.exception.code, "media_timeout")
        self.assertEqual(process.terminated, 1)
        self.assert_safe(caught.exception)

    async def test_process_that_ignores_terminate_is_killed_after_grace_period(self):
        class Process:
            def __init__(self):
                self.returncode = None
                self.terminated = 0
                self.killed = 0
                self.ended = asyncio.Event()
            async def wait(self):
                await self.ended.wait()
                return self.returncode
            def terminate(self):
                self.terminated += 1
            def kill(self):
                self.killed += 1
                self.returncode = -9
                self.ended.set()
        process = Process()
        with patch.object(remote, "PROCESS_STOP_SECONDS", 0.01):
            await asyncio.wait_for(remote._terminate_owned_process(process), 1)
        self.assertEqual(process.terminated, 1)
        self.assertEqual(process.killed, 1)

    async def test_spawn_error_is_sanitized(self):
        with patch.object(remote.asyncio, "create_subprocess_exec", AsyncMock(side_effect=OSError(FAKE_KEY + SECRET_URL))):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                await remote._run_process(["ffmpeg"], None)
        self.assertEqual(caught.exception.code, "media_tool_unavailable")
        self.assert_safe(caught.exception)


class RemoteEngineTests(SafeAssertions, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="engine-", dir=TEST_ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.temp = self.root / "temp"
        self.temp.mkdir()
        self.source = self.root / "个人 视频 with spaces.mp4"
        self.source.write_bytes(ORIGINAL_MEDIA)
        self.config = {"url": SECRET_URL + "/audio/transcriptions", "model": "whisper-1",
                       "api_key": FAKE_KEY, "proxy": {"enabled": False, "url": "", "port": 8080}}
        self.requests = []
        self.client_options = []
        self.destinations = []
        self.durations = []
        self.probe = AsyncMock(return_value=12.0)
        self.enterContext(patch.object(remote, "_probe_duration", self.probe))
        self.enterContext(patch.object(remote.lvats_paths, "TEMP_DIR", self.temp))
        self.enterContext(patch.object(remote, "CANCEL_POLL_SECONDS", 0.001))
        self.enterContext(patch.object(remote, "_extract_chunk", AsyncMock(side_effect=self.extract)))

    async def extract(self, source, destination, offset, duration, cancel):
        self.assertEqual(source, self.source.resolve())
        self.assertEqual(destination.name, "audio.mp3")
        self.assertEqual(destination.parent.parent, self.temp)
        self.assertTrue(destination.parent.name.startswith("remote-api-"))
        self.assertFalse(destination.exists(), "Previous audio block must be removed before extracting next")
        self.destinations.append(destination)
        self.durations.append((offset, duration))
        destination.write_bytes(EXTRACTED_AUDIO)

    def mock_client(self, handler):
        async def wrapped(request):
            await request.aread()
            self.requests.append(request)
            result = handler(request)
            if hasattr(result, "__await__"):
                result = await result
            return result

        def factory(**kwargs):
            self.client_options.append(dict(kwargs))
            # The real proxy route is tested separately with a loopback server.
            kwargs.pop("proxy", None)
            return REAL_CLIENT(transport=httpx.MockTransport(wrapped), **kwargs)

        return patch.object(remote.httpx, "AsyncClient", side_effect=factory)

    def assert_clean(self):
        self.assertEqual(list(self.temp.iterdir()), [])
        for destination in self.destinations:
            self.assertFalse(destination.exists())
        self.assertEqual(self.source.read_bytes(), ORIGINAL_MEDIA)

    async def test_txt_success_multipart_private_filename_and_secure_options(self):
        reports = []
        with self.mock_client(lambda request: httpx.Response(200, json={"text": "识别成功"})):
            result = await remote.transcribe(self.source, self.config, prompt="术语提示", progress_cb=lambda *args: reports.append(args))
        self.assertEqual(result, {"text": "识别成功", "segments": [], "cancelled": False, "degraded": False})
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        body = request.content
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), self.config["url"])
        self.assertEqual(request.headers["authorization"], "Bearer " + FAKE_KEY)
        self.assertIn(b'filename="audio.mp3"', body)
        self.assertIn(EXTRACTED_AUDIO, body)
        self.assertNotIn(ORIGINAL_MEDIA, body)
        self.assertNotIn(str(self.source).encode("utf-8"), body)
        self.assertNotIn(self.source.name.encode("utf-8"), body)
        self.assertNotIn(FAKE_KEY.encode(), body)
        self.assertIn(b'name="response_format"\r\n\r\njson', body)
        self.assertIn("术语提示".encode(), body)
        options = self.client_options[0]
        self.assertIsNone(options["proxy"])
        self.assertFalse(options["trust_env"])
        self.assertFalse(options["follow_redirects"])
        self.assertTrue(options["verify"])
        self.assertEqual(options["timeout"].connect, 20)
        self.assertEqual(options["timeout"].read, 180)
        self.assertEqual(remote.REQUEST_TOTAL_SECONDS, 300)
        self.assertEqual(reports[-1][0], 100)
        self.assertNotIn(FAKE_KEY, repr(reports))
        self.assertNotIn(SECRET_URL, repr(reports))
        self.assert_clean()

    async def test_two_chunks_verbose_json_and_offset_merge(self):
        self.probe.return_value = 605.0
        responses = [{"text": "第一块", "segments": [{"start": 1, "end": 4, "text": "第一块"}]},
                     {"text": "第二块", "segments": [{"start": 0.5, "end": 4, "text": "第二块"}]}]
        with self.mock_client(lambda request: httpx.Response(200, json=responses.pop(0))):
            result = await remote.transcribe(self.source, self.config, need_timestamps=True)
        self.assertEqual(result["text"], "第一块\n第二块")
        self.assertEqual(result["segments"], [{"start": 1.0, "end": 4.0, "text": "第一块"},
                                             {"start": 600.5, "end": 604.0, "text": "第二块"}])
        self.assertEqual(self.durations, [(0.0, 600.0), (600.0, 5.0)])
        self.assertIn("warning", result)
        self.assertFalse(result["degraded"])
        for request in self.requests:
            self.assertIn(b'verbose_json', request.content)
            self.assertIn(b'name="timestamp_granularities[]"\r\n\r\nsegment', request.content)
        self.assert_clean()

    async def test_proxy_option_is_explicit_and_environment_is_ignored(self):
        self.config["proxy"] = {"enabled": True, "url": "http://127.0.0.1", "port": 3128}
        with patch.dict(os.environ, {"HTTP_PROXY": "http://bad.invalid:1234", "HTTPS_PROXY": "http://bad.invalid:1234"}), \
                self.mock_client(lambda request: httpx.Response(200, json={"text": "ok"})):
            await remote.transcribe(self.source, self.config)
        self.assertEqual(self.client_options[0]["proxy"], "http://127.0.0.1:3128")
        self.assertFalse(self.client_options[0]["trust_env"])
        self.assert_clean()

    async def test_http_errors_classified_no_leaks_retries_or_redirect_following(self):
        errors = {302: "redirect_blocked", 307: "redirect_blocked", 401: "authentication_failed",
                  403: "authentication_failed", 407: "proxy_authentication_required", 413: "upload_too_large",
                  429: "rate_limited", 400: "protocol_unsupported", 404: "protocol_unsupported",
                  500: "service_error", 504: "timeout"}
        for status, code in errors.items():
            self.requests.clear()
            with self.subTest(status=status), self.mock_client(lambda request: httpx.Response(
                    status, content=(FAKE_KEY + SECRET_URL).encode(), headers={"location": "https://other.invalid/"})):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    await remote.transcribe(self.source, self.config)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.http_status, status)
            self.assertEqual(len(self.requests), 1)
            self.assert_safe(caught.exception, self.source)
            self.assert_clean()

    async def test_network_timeout_and_proxy_errors_are_sanitized(self):
        cases = [(httpx.ReadTimeout, "timeout", False), (httpx.WriteTimeout, "timeout", False),
                 (httpx.ConnectError, "network_error", False), (httpx.ProxyError, "proxy_unavailable", True),
                 (httpx.ConnectError, "proxy_unavailable", True), (httpx.RemoteProtocolError, "network_error", False)]
        for kind, code, proxy in cases:
            self.config["proxy"] = {"enabled": proxy, "url": "http://127.0.0.1", "port": 3128}
            self.requests.clear()
            def fail(request):
                raise kind(FAKE_KEY + SECRET_URL, request=request)
            with self.subTest(kind=kind.__name__), self.mock_client(fail):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    await remote.transcribe(self.source, self.config)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(len(self.requests), 1)
            self.assert_safe(caught.exception)
            self.assert_clean()

    async def test_unexpected_transport_exception_is_sanitized(self):
        def fail(request):
            raise RuntimeError(FAKE_KEY + SECRET_URL + str(self.source))
        with self.mock_client(fail):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                await remote.transcribe(self.source, self.config)
        self.assertEqual(caught.exception.code, "remote_internal_error")
        self.assert_safe(caught.exception, self.source)
        self.assertEqual(len(self.requests), 1)
        self.assert_clean()

    async def test_total_request_deadline_stops_slow_service_without_retry(self):
        stopped = asyncio.Event()
        async def pending(request):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        with patch.object(remote, "REQUEST_TOTAL_SECONDS", 0.01), self.mock_client(pending):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                await remote.transcribe(self.source, self.config)
        self.assertEqual(caught.exception.code, "timeout")
        self.assert_safe(caught.exception)
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(stopped.is_set())
        self.assert_clean()

    async def test_upload_limit_prevents_request(self):
        with patch.object(remote, "MAX_UPLOAD_BYTES", len(EXTRACTED_AUDIO)), \
                self.mock_client(lambda request: self.fail("Oversized audio must not be sent")):
            with self.assertRaises(remote.RemoteAPIError) as caught:
                await remote.transcribe(self.source, self.config)
        self.assertEqual(caught.exception.code, "upload_too_large")
        self.assertEqual(self.requests, [])
        self.assert_clean()

    async def test_response_size_header_stream_and_compression_limits(self):
        class Pieces(httpx.AsyncByteStream):
            async def __aiter__(self):
                for _ in range(3):
                    yield b"x" * 8

        responses = [httpx.Response(200, content=b"{}", headers={"content-length": "1000"}),
                     httpx.Response(200, stream=Pieces()),
                     httpx.Response(200, content=b"{}", headers={"content-length": "bad"}),
                     httpx.Response(200, stream=Pieces(), headers={"content-encoding": "gzip"})]
        for response in responses:
            with self.subTest(headers=list(response.headers)), patch.object(remote, "MAX_RESPONSE_BYTES", 10), \
                    self.mock_client(lambda request: response):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    await remote.transcribe(self.source, self.config)
            self.assertIn(caught.exception.code, {"response_too_large", "invalid_response"})
            self.assert_safe(caught.exception)
            self.assert_clean()

    async def test_bad_protocol_and_subtitle_failure_never_fallback(self):
        for data, timestamps, code in [(b"garbage " + FAKE_KEY.encode(), False, "invalid_response"),
                                       (b'{"text":"complete text"}', True, "timestamps_unsupported")]:
            with self.mock_client(lambda request: httpx.Response(200, content=data)):
                with self.assertRaises(remote.RemoteAPIError) as caught:
                    await remote.transcribe(self.source, self.config, need_timestamps=timestamps)
            self.assertEqual(caught.exception.code, code)
            self.assert_safe(caught.exception)
            self.assert_clean()

    async def test_cancel_before_processing_never_probes_or_contacts_service(self):
        cancel = threading.Event()
        cancel.set()
        with self.mock_client(lambda request: self.fail("Cancelled job must not upload")):
            result = await remote.transcribe(self.source, self.config, cancel=cancel)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["text"], "")
        self.probe.assert_not_called()
        self.assertEqual(self.client_options, [])
        self.assert_clean()

    async def test_cancel_during_extraction_sends_no_audio_and_cleans_temp(self):
        started = asyncio.Event()
        stopped = asyncio.Event()
        cancel = threading.Event()
        async def extracting(source, destination, offset, duration, token):
            destination.write_bytes(EXTRACTED_AUDIO)
            started.set()
            try:
                await remote._await_cancel(asyncio.Event().wait(), token)
            finally:
                stopped.set()
        with patch.object(remote, "_extract_chunk", AsyncMock(side_effect=extracting)), \
                self.mock_client(lambda request: self.fail("Cancelled extraction must not upload")):
            task = asyncio.create_task(remote.transcribe(self.source, self.config, cancel=cancel))
            await asyncio.wait_for(started.wait(), 1)
            cancel.set()
            result = await asyncio.wait_for(task, 1)
        self.assertTrue(result["cancelled"])
        self.assertTrue(stopped.is_set())
        self.assertEqual(self.requests, [])
        self.assert_clean()

    async def test_cancel_callback_at_upload_boundary_prevents_request(self):
        cancel = threading.Event()
        def progress(percent, message, partial=None, meta=None):
            if 0 < percent < 90:
                cancel.set()
        with self.mock_client(lambda request: self.fail("Cancelled job must not upload")):
            result = await remote.transcribe(self.source, self.config, cancel=cancel, progress_cb=progress)
        self.assertTrue(result["cancelled"])
        self.assertEqual(self.requests, [])
        self.assert_clean()

    async def test_cancel_active_http_and_external_task_cancel_clean_owned_work(self):
        for external in (False, True):
            started = asyncio.Event()
            stopped = asyncio.Event()
            cancel = threading.Event()
            async def pending(request):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
            with self.subTest(external=external), self.mock_client(pending):
                task = asyncio.create_task(remote.transcribe(self.source, self.config, cancel=cancel))
                await asyncio.wait_for(started.wait(), 1)
                if external:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    cancel.set()
                    result = await asyncio.wait_for(task, 1)
                    self.assertTrue(result["cancelled"])
                    self.assertEqual(result["segments"], [])
                self.assertTrue(stopped.is_set())
            self.assert_clean()

    async def test_cancel_active_response_stream_closes_stream(self):
        started = asyncio.Event()
        closed = asyncio.Event()
        cancel = threading.Event()
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                yield b'{"text":"'
                await asyncio.Event().wait()
            async def aclose(self):
                closed.set()
        with self.mock_client(lambda request: httpx.Response(200, stream=SlowStream())):
            task = asyncio.create_task(remote.transcribe(self.source, self.config, cancel=cancel))
            await asyncio.wait_for(started.wait(), 1)
            cancel.set()
            result = await asyncio.wait_for(task, 1)
        self.assertTrue(result["cancelled"])
        self.assertTrue(closed.is_set())
        self.assert_clean()

    async def test_cancel_between_chunks_preserves_partial_only(self):
        self.probe.return_value = 605
        cancel = threading.Event()
        def progress(percent, message, partial=None, meta=None):
            if meta and meta.get("timing", {}).get("chunk") == 1:
                cancel.set()
        with self.mock_client(lambda request: httpx.Response(200, json={"text": "first"})):
            result = await remote.transcribe(self.source, self.config, progress_cb=progress, cancel=cancel)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["text"], "first")
        self.assertEqual(len(self.requests), 1)
        self.assert_clean()

    async def test_progress_failure_and_async_callbacks_do_not_break_engine(self):
        async def bad_progress(*args):
            raise RuntimeError(FAKE_KEY)
        with self.mock_client(lambda request: httpx.Response(200, json={"text": "ok"})):
            result = await remote.transcribe(self.source, self.config, progress_cb=bad_progress)
        self.assertEqual(result["text"], "ok")
        self.assert_clean()

    async def test_invalid_runtime_prompt_source_and_duration_do_not_upload(self):
        with self.mock_client(lambda request: self.fail("Invalid job must not upload")):
            for configuration in [{}, {**self.config, "api_key": "\r\nsecret"},
                                  {**self.config, "url": "http://nonloopback.example.test/v1"}]:
                with self.assertRaises(remote.RemoteAPIError):
                    await remote.transcribe(self.source, configuration)
            with self.assertRaises(remote.RemoteAPIError) as caught:
                await remote.transcribe(self.source, self.config, prompt="x" * 8001)
            self.assertEqual(caught.exception.code, "invalid_prompt")
            with self.assertRaises(remote.RemoteAPIError) as caught:
                await remote.transcribe(self.root / "missing file", self.config)
            self.assertEqual(caught.exception.code, "invalid_media")
            for duration in [float("nan"), -1, 700000]:
                self.probe.return_value = duration
                with self.assertRaises(remote.RemoteAPIError):
                    await remote.transcribe(self.source, self.config)
        self.assertEqual(self.requests, [])
        self.assert_clean()

    async def test_parallel_transcriptions_have_distinct_temporary_workspaces(self):
        rendezvous = asyncio.Event()
        count = 0
        async def respond(request):
            nonlocal count
            count += 1
            if count == 2:
                rendezvous.set()
            await rendezvous.wait()
            return httpx.Response(200, json={"text": "ok"})
        with self.mock_client(respond):
            results = await asyncio.wait_for(asyncio.gather(
                remote.transcribe(self.source, self.config), remote.transcribe(self.source, self.config)), 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(len({path.parent for path in self.destinations}), 2)
        self.assert_clean()

    async def test_runtime_is_copied_before_await_and_not_redirected_to_new_settings(self):
        original = dict(self.config)
        async def probe(source, cancel):
            self.config["url"] = "https://changed-endpoint.example.test/audio/transcriptions"
            self.config["api_key"] = "not-real-changed-token"
            return 12
        self.probe.side_effect = probe
        with self.mock_client(lambda request: httpx.Response(200, json={"text": "ok"})):
            await remote.transcribe(self.source, self.config)
        self.assertEqual(str(self.requests[0].url), original["url"])
        self.assertEqual(self.requests[0].headers["authorization"], "Bearer " + original["api_key"])
        self.assert_clean()

    async def test_real_loopback_direct_and_http_proxy_routing(self):
        captured = []
        connections = set()
        async def serve(reader, writer):
            connections.add(writer)
            try:
                header = await reader.readuntil(b"\r\n\r\n")
                size = next(int(line.split(b":", 1)[1].strip()) for line in header.split(b"\r\n")
                            if line.lower().startswith(b"content-length:"))
                body = await reader.readexactly(size)
                captured.append((header, body))
                payload = b'{"text":"loopback-ok"}'
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                             + str(len(payload)).encode() + b"\r\nConnection: close\r\n\r\n" + payload)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                connections.discard(writer)

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            self.config["url"] = f"http://127.0.0.1:{port}/v1/audio/transcriptions"
            with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": "*"}):
                direct = await remote.transcribe(self.source, self.config)
            self.assertEqual(direct["text"], "loopback-ok")
            # 127.0.0.2:9 is unreachable in the test: successful response proves
            # the explicit proxy received the absolute-form request, not origin.
            self.config["url"] = "http://127.0.0.2:9/v1/audio/transcriptions"
            self.config["proxy"] = {"enabled": True, "url": "http://127.0.0.1", "port": port}
            with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": "*"}):
                proxied = await remote.transcribe(self.source, self.config)
            self.assertEqual(proxied["text"], "loopback-ok")
        finally:
            server.close()
            await server.wait_closed()
            for writer in list(connections):
                writer.close()
                await writer.wait_closed()
        self.assertEqual(len(captured), 2)
        self.assertTrue(captured[0][0].startswith(b"POST /v1/audio/transcriptions HTTP/1.1"))
        self.assertTrue(captured[1][0].startswith(b"POST http://127.0.0.2:9/v1/audio/transcriptions HTTP/1.1"))
        for header, body in captured:
            self.assertIn(("Authorization: Bearer " + FAKE_KEY).encode(), header)
            self.assertIn(EXTRACTED_AUDIO, body)
            self.assertNotIn(ORIGINAL_MEDIA, body)
            self.assertNotIn(self.source.name.encode(), body)
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
