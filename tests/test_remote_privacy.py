"""Provider success payloads must not carry the current API key into outputs or SSE."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

os.environ['ASR_TEST_MODE'] = '1'
os.environ['ASR_PLACEHOLDER'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import remote_api


class ProviderPrivacy(unittest.IsolatedAsyncioTestCase):
    async def test_successful_provider_key_echo_fails_closed(self):
        key = 'privacy-fixture-not-a-real-key'
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'audio.mp3'
            audio.write_bytes(b'fixture')
            for needs_timestamps in (False, True):
                payload = {'text': key, 'segments': [{'start': 0, 'end': 0.5, 'text': key}]}
                transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
                async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
                    with self.assertRaises(remote_api.RemoteAPIError) as caught:
                        await remote_api._send_chunk(client, audio,
                            {'url': 'https://fixture.example/v1/audio/transcriptions', 'model': 'whisper-1', 'api_key': key},
                            None, 1.0, 0.0, needs_timestamps)
                    self.assertNotIn(key, str(caught.exception))

    async def test_subtitle_empty_lines_do_not_terminate_cues(self):
        content = json.dumps({'text': 'first\n\nsecond', 'segments': [
            {'start': 0, 'end': 1, 'text': 'first\n\nsecond'}]}).encode()
        text, segments = remote_api._parse_response(content, 2, 0, True)
        self.assertEqual(text, 'first\n\nsecond')
        self.assertEqual(segments[0]['text'], 'first second')

    async def test_aggregate_result_cap_and_bounded_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'fixture.wav'
            source.write_bytes(b'fixture')
            async def extract(source, destination, *args):
                destination.write_bytes(b'fixture audio')
            config = {'url': 'https://fixture.example/v1', 'model': 'whisper-1',
                      'api_key': 'privacy-fixture-not-a-real-key',
                      'proxy': {'enabled': False, 'url': '', 'port': 8080}}
            callbacks = []
            def callback(pct, message, partial=None, meta=None):
                if partial is not None:
                    callbacks.append(partial)
            with patch.object(remote_api.lvats_paths, 'TEMP_DIR', root), \
                 patch.object(remote_api, '_probe_duration', new_callable=AsyncMock, return_value=1200), \
                 patch.object(remote_api, '_extract_chunk', side_effect=extract), \
                 patch.object(remote_api, '_send_chunk', new_callable=AsyncMock, return_value=('abcdef', [])), \
                 patch.object(remote_api, 'MAX_TASK_TEXT_BYTES', 10):
                with self.assertRaises(remote_api.RemoteAPIError) as caught:
                    await remote_api.transcribe(source, config, progress_cb=callback)
                self.assertEqual(caught.exception.code, 'response_too_large')
            callbacks.clear()
            with patch.object(remote_api.lvats_paths, 'TEMP_DIR', root), \
                 patch.object(remote_api, '_probe_duration', new_callable=AsyncMock, return_value=1200), \
                 patch.object(remote_api, '_extract_chunk', side_effect=extract), \
                 patch.object(remote_api, '_send_chunk', new_callable=AsyncMock, return_value=('abcdef', [])), \
                 patch.object(remote_api, 'MAX_PROGRESS_CHARS', 4):
                result = await remote_api.transcribe(source, config, progress_cb=callback)
                self.assertEqual(result['text'], 'abcdef\nabcdef')
                self.assertTrue(all(len(value) <= 4 for value in callbacks))
            self.assertFalse(list(root.glob('remote-api-*')))


if __name__ == '__main__':
    unittest.main()
