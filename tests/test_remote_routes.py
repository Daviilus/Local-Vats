"""Local API configuration/task integration; fake keys, isolated data, no speech models."""
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
os.environ['ASR_PORT'] = '8001'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import remote_api
import server


class TestProtector:
    def protect(self, value):
        return b'test:' + bytes(byte ^ 0x63 for byte in value)

    def unprotect(self, value):
        return bytes(byte ^ 0x63 for byte in value[5:])


class RemoteRoutes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = remote_api.ConfigStore(self.root / 'remote.test.json', protector=TestProtector())
        self.patches = [
            patch.object(server, '_remote_store', return_value=self.store),
            patch.object(server, 'TASKS', {}), patch.object(server, 'FILES', {}),
            patch.object(server, 'TASK_QUEUE', server.TaskQueue()),
            patch.object(server, 'BUS', server.EventBus()),
            patch.object(server, '_shutdown_started', False),
            patch.object(server, 'QUEUE_FILE', self.root / 'queue.test.json'),
            patch.object(server, 'STATE_DIR', self.root),
        ]
        for item in self.patches:
            item.start()
        self.transport = httpx.ASGITransport(app=server.app, client=('127.0.0.1', 40000))
        self.client = httpx.AsyncClient(transport=self.transport, base_url='http://127.0.0.1:8001')
        self.headers = {'X-Lvats-Request': '1', 'Origin': 'http://127.0.0.1:8001'}
        self.key = 'test-only-not-a-real-api-key'
        self.settings = {'url': 'https://asr.example/v1', 'model': 'whisper-1', 'api_key': self.key,
                         'proxy': {'enabled': False, 'url': '', 'port': 8080}}

    async def asyncTearDown(self):
        await self.client.aclose()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    async def put_config(self, payload):
        return await self.client.put('/api/remote/config', headers=self.headers,
                                     json={**payload, 'expected_revision': self.store.public()['revision']})

    async def apply(self):
        response = await self.put_config(self.settings)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn(self.key, response.text)
        return response.json()

    async def test_configuration_is_secret_free_and_no_store(self):
        applied = await self.apply()
        self.assertTrue(applied['ready'])
        response = await self.client.get('/api/remote/config', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(response.headers['x-frame-options'], 'DENY')
        self.assertNotIn(self.key, response.text)
        self.assertNotIn('api_key_encrypted', response.text)
        self.assertNotIn(self.key, self.store.path.read_text(encoding='utf-8'))
        kept = await self.put_config({'model': 'different-model', 'api_key': ''})
        self.assertTrue(kept.json()['api_key_set'])
        self.assertNotEqual(applied['revision'], kept.json()['revision'])
        cleared = await self.put_config({'clear_api_key': True})
        self.assertFalse(cleared.json()['api_key_set'])
        self.assertFalse(cleared.json()['ready'])

    async def test_cross_origin_host_and_missing_header_rejected(self):
        for headers in ({}, {'X-Lvats-Request': '1', 'Origin': 'https://unrelated.example'},
                        {'X-Lvats-Request': '1', 'Sec-Fetch-Site': 'cross-site'},
                        {'X-Lvats-Request': '1', 'Host': 'attacker.example'}):
            response = await self.client.put('/api/remote/config', headers=headers, json=self.settings)
            self.assertEqual(response.status_code, 403)
            self.assertNotIn(self.key, response.text)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=('192.0.2.1', 40000)),
                                   base_url='http://127.0.0.1:8001')
        async with client:
            response = await client.get('/api/remote/config', headers=self.headers)
            self.assertEqual(response.status_code, 403)
        self.assertFalse(self.store.public()['ready'])

    async def test_invalid_body_cannot_echo_keys(self):
        bodies = [json.dumps({'expected_revision': self.store.public()['revision'], 'api_key': self.key, 'proxy': {'enabled': True, 'url': 'http://127.0.0.1', 'port': 0}}),
                  '{"api_key":"' + self.key + '"', json.dumps([{'api_key': self.key}])]
        for body in bodies:
            response = await self.client.put('/api/remote/config', headers={**self.headers, 'Content-Type': 'application/json'}, content=body)
            self.assertEqual(response.status_code, 400)
            self.assertNotIn(self.key, response.text)
        response = await self.client.put('/api/remote/config', headers={**self.headers, 'Content-Type': 'application/json'},
                                         content=json.dumps({'api_key': self.key + 'x' * 20000}))
        self.assertEqual(response.status_code, 413)
        self.assertNotIn(self.key, response.text)

    async def test_configured_api_is_explicit_option_not_local_default(self):
        with patch.object(server.model_registry, 'all_models', return_value=[]):
            self.assertEqual((await self.client.get('/api/models')).json()['models'], [])
            await self.apply()
            models = (await self.client.get('/api/models')).json()['models']
            self.assertEqual(models[0]['id'], remote_api.REMOTE_MODEL_ID)
            self.assertEqual(models[0]['engine'], 'remote-api')
        with patch.object(server.asr_model, 'load_model', side_effect=AssertionError('local model load')):
            result = await server.api_preload({'model_id': remote_api.REMOTE_MODEL_ID})
        self.assertTrue(result['skipped'])

    async def add_task(self, consent=True):
        media = self.root / 'sample.wav'
        media.write_bytes(b'sample fixture')
        server.FILES['fixture'] = {'id': 'fixture', 'path': str(media), 'name': media.name, 'ok': True}
        return await self.client.post('/api/tasks/add', headers=self.headers,
                                      json={'file_ids': ['fixture'], 'format': 'srt', 'model_id': remote_api.REMOTE_MODEL_ID,
                                            'remote_consent': consent, 'remote_revision': self.store.public()['revision']})

    async def test_api_tasks_require_consent_and_persist_only_identity(self):
        response = await self.add_task()
        self.assertEqual(response.status_code, 400)
        await self.apply()
        response = await self.add_task(False)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(server.TASKS)
        response = await self.add_task()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['created'], 1)
        task = next(iter(server.TASKS.values()))
        self.assertEqual(task['remote_revision'], self.store.public()['revision'])
        saved = server.QUEUE_FILE.read_text(encoding='utf-8')
        self.assertNotIn(self.key, saved)
        self.assertNotIn(self.settings['url'], saved)
        self.assertNotIn('api_key', saved)
        state = (await self.client.get('/api/state')).json()
        self.assertEqual(state['tasks'][0]['model_label'], 'API · whisper-1')

    async def test_revision_mismatch_fails_before_media_dispatch(self):
        await self.apply()
        self.assertEqual((await self.add_task()).status_code, 200)
        task = next(iter(server.TASKS.values()))
        self.store.update({'model': 'changed'})
        with patch.object(remote_api, 'transcribe', new_callable=AsyncMock) as transcribe, \
             patch.object(server.model_registry, 'resolve', side_effect=AssertionError('local fallback')):
            with self.assertRaises(remote_api.RemoteAPIError):
                await server._run_engine(task, task['id'], server.CancelToken(), None)
            transcribe.assert_not_awaited()

    async def test_dispatch_uses_private_snapshot_without_local_model(self):
        await self.apply()
        await self.add_task()
        task = next(iter(server.TASKS.values()))
        task['status'] = 'running'
        expected = {'text': 'fixture text', 'segments': [{'start': 0, 'end': 1, 'text': 'fixture text'}], 'cancelled': False}
        with patch.object(remote_api, 'transcribe', new_callable=AsyncMock, return_value=expected) as transcribe, \
             patch.object(server.model_registry, 'resolve', side_effect=AssertionError('local fallback')):
            self.assertEqual(await server._run_engine(task, task['id'], server.CancelToken(), None), expected)
            self.assertEqual(transcribe.call_args.args[1]['api_key'], self.key)
            self.assertTrue(transcribe.call_args.kwargs['need_timestamps'])
            self.assertEqual(json.loads(server.QUEUE_FILE.read_text())['tasks'][0]['status'], 'running')
        self.assertNotIn(self.key, json.dumps(server._serializable(task)))

    async def test_api_running_state_must_persist_before_upload(self):
        await self.apply()
        await self.add_task()
        task = next(iter(server.TASKS.values()))
        task['status'] = 'running'
        with patch.object(server, 'persist_now', side_effect=RuntimeError('fixture failure')) as persist, \
             patch.object(remote_api, 'transcribe', new_callable=AsyncMock) as transcribe:
            with self.assertRaises(RuntimeError) as caught:
                await server._run_engine(task, task['id'], server.CancelToken(), None)
            self.assertIn('未发送媒体', str(caught.exception))
            persist.assert_called_once_with(strict=True)
            transcribe.assert_not_awaited()

    async def test_interrupted_remote_task_is_not_automatically_reuploaded(self):
        await self.apply()
        await self.add_task()
        task = next(iter(server.TASKS.values()))
        task['status'] = 'running'
        task['started_at'] = 1.0
        server.persist_now()
        server.TASKS.clear()
        server.TASK_QUEUE.load([])
        self.assertEqual(server.restore_queue(), 1)
        restored = next(iter(server.TASKS.values()))
        self.assertEqual(restored['status'], 'failed')
        self.assertIn('不会自动重新上传', restored['error'])
        self.assertEqual(server.TASK_QUEUE.snapshot(), [])

    async def test_shutdown_blocks_new_tasks_and_paid_dequeue(self):
        await self.apply()
        await self.add_task()
        saved_order = server.TASK_QUEUE.snapshot()
        server._shutdown_started = True
        rejected = await self.add_task()
        self.assertEqual(rejected.status_code, 409)
        with patch.object(remote_api, 'transcribe', new_callable=AsyncMock) as transcribe:
            await server._worker()
            transcribe.assert_not_awaited()
        self.assertEqual(server.TASK_QUEUE.snapshot(), saved_order)

    async def test_cleanup_wait_has_bounded_window_for_owned_operations(self):
        import threading
        event = threading.Event()
        with patch.object(server, '_REMOTE_OPERATIONS', {event}):
            self.assertFalse(server._wait_remote_cleanup(0.01))
            event.set()
            self.assertTrue(server._wait_remote_cleanup(0.01))

    async def test_stale_form_or_task_cannot_select_changed_provider(self):
        applied = await self.apply()
        self.store.update({'url': 'https://different.example/v1'})
        stale = await self.client.put('/api/remote/config', headers=self.headers,
                                      json={'expected_revision': applied['revision'], 'url': self.settings['url']})
        self.assertEqual(stale.status_code, 409)
        media = self.root / 'fixture.wav'
        media.write_bytes(b'fixture')
        server.FILES['fixture'] = {'id': 'fixture', 'path': str(media), 'name': media.name, 'ok': True}
        task = await self.client.post('/api/tasks/add', headers=self.headers,
                                     json={'file_ids': ['fixture'], 'format': 'txt', 'model_id': remote_api.REMOTE_MODEL_ID,
                                           'remote_consent': True, 'remote_revision': applied['revision']})
        self.assertEqual(task.status_code, 409)
        self.assertFalse(server.TASKS)

    async def test_config_page_has_navigation_and_security_headers(self):
        response = await self.client.get('/api-settings')
        self.assertEqual(response.status_code, 200)
        self.assertIn('API 代理配置', response.text)
        self.assertIn('frame-ancestors', response.headers['content-security-policy'])
        self.assertIn('no-store', response.headers['cache-control'])
        main = await self.client.get('/')
        self.assertIn('href="/api-settings"', main.text)
        self.assertIn('remote_consent', main.text)


if __name__ == '__main__':
    unittest.main()
