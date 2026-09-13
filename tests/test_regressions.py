"""No real model loading. Run: python -m unittest discover -s tests -v."""
import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

os.environ['ASR_TEST_MODE'] = '1'
os.environ['ASR_PLACEHOLDER'] = '1'
os.environ['ASR_PORT'] = '8001'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asr_model as asr
import fw_engine as fw
import model_registry as registry
import server
from starlette.requests import Request

BASE_DIR = Path(__file__).resolve().parents[1]


class Regressions(unittest.TestCase):
    def setUp(self):
        self.model_tmp = tempfile.TemporaryDirectory()
        model_root = Path(self.model_tmp.name) / 'models'
        model_root.mkdir()
        for name in ('Qwen3-ASR-0.6B-hf', 'Qwen3-ASR-1.7B-hf'):
            directory = model_root / name
            directory.mkdir()
            (directory / 'config.json').write_text(json.dumps({
                'architectures': ['Qwen3ASRForConditionalGeneration'],
            }), encoding='utf-8')
            (directory / 'model.safetensors').touch()
        whisper = model_root / 'faster-whisper-tiny'
        whisper.mkdir()
        (whisper / 'model.bin').touch()
        (whisper / 'tokenizer.json').write_text('{}', encoding='utf-8')
        self.models_patch = patch.object(registry, 'MODELS_DIR', model_root)
        self.qwen_patch = patch.object(registry, 'QWEN_DIR_CANDIDATES', [])
        self.models_patch.start()
        self.qwen_patch.start()
        server.BUS = server.EventBus()
        server.FILES.clear()
        server.TASKS.clear()
        server.DOWNLOAD_JOBS.clear()
        server._shutdown_started = False
        registry.load()

    def tearDown(self):
        asr._model = asr._processor = asr._model_path = None
        asr._device_info = {}
        fw._model = fw._model_id = fw._model_path = None
        server._preload_task = None
        self.qwen_patch.stop()
        self.models_patch.stop()
        self.model_tmp.cleanup()

    def test_builtin_download_list_contains_only_qwen_asr_models(self):
        self.assertEqual(
            [item['repo_id'] for item in registry.BUILTIN_DOWNLOADS],
            ['Qwen/Qwen3-ASR-0.6B-hf', 'Qwen/Qwen3-ASR-1.7B-hf'],
        )

    def test_incomplete_qwen_download_is_not_registered(self):
        partial = registry.MODELS_DIR / 'Qwen3-ASR-partial'
        partial.mkdir()
        (partial / 'config.json').write_text(json.dumps({
            'architectures': ['Qwen3ASRForConditionalGeneration'],
        }), encoding='utf-8')
        self.assertNotIn(partial.name, registry.scan())

    def test_first_start_queues_two_qwen_models_and_aligner(self):
        registry._REGISTRY = {}
        with patch.object(asr, 'find_aligner_dir', return_value=None), \
             patch.object(server.threading, 'Thread') as thread, \
             patch.object(server, '_log'):
            server._queue_initial_model_downloads()
        self.assertEqual(
            {job['repo_id'] for job in server.DOWNLOAD_JOBS.values()},
            {
                'Qwen/Qwen3-ASR-0.6B-hf',
                'Qwen/Qwen3-ASR-1.7B-hf',
                'Qwen/Qwen3-ForcedAligner-0.6B-hf',
            },
        )
        self.assertEqual(thread.call_count, 1)

    def test_selected_qwen_path_reaches_transcription(self):
        entry = registry.get('Qwen3-ASR-0.6B-hf')
        task = dict(model_id=entry['id'], path='sample.wav', format='srt')
        with patch.object(asr, 'transcribe_audio', return_value={}) as transcribe:
            asyncio.run(server._run_engine(task, 'test', None, None))
        self.assertEqual(transcribe.call_args.kwargs['model_dir'], entry['path'])

    def test_qwen_switch_releases_old_model_before_loading(self):
        calls = []
        fake = MagicMock()
        fake.AutoProcessor.from_pretrained.side_effect = lambda p: calls.append(('processor', p)) or object()
        def create(loader, path):
            self.assertIsNone(asr._model, 'old GPU model must be released first')
            calls.append(('model', path))
            return MagicMock()
        with patch.dict(sys.modules, {'transformers': fake}), \
             patch.object(asr, '_load_torch'), patch.object(asr, '_load_quantized', side_effect=create), \
             patch.object(asr, '_compute_device_info'), patch.object(asr, '_model', None), \
             patch.object(asr, '_processor', None), patch.object(asr, '_model_path', None, create=True):
            small = registry.get('Qwen3-ASR-0.6B-hf')['path']
            large = registry.get('Qwen3-ASR-1.7B-hf')['path']
            asr.load_model(small)
            asr.load_model(small)
            asr.load_model(large)
            self.assertEqual([p for k, p in calls if k == 'model'], [small, large])

    def test_placeholder_whisper_never_loads_real_model(self):
        entry = registry.get('faster-whisper-tiny')
        with patch.object(fw, '_load', side_effect=AssertionError('real load forbidden')):
            result = fw.transcribe(Path('sample.wav'), entry)
        self.assertTrue(result['text'])
        self.assertTrue(result['segments'])

    def test_test_paths_are_isolated(self):
        self.assertNotEqual(server.PID_FILE.name, 'lvats.pid')
        self.assertNotEqual(server.QUICK_PROMPTS_FILE.name, 'quick_prompts.json')
        self.assertNotEqual(registry.REGISTRY_FILE.name, 'registry.json')
        self.assertNotEqual(asr.CACHE_DIR.name, '.wavcache')
        self.assertNotEqual(server.UPLOAD_DIR.name, '.dropcache')

    def test_file_queue_enforces_fifty_item_limit(self):
        paths = [f'C:/fixtures/{index}.wav' for index in range(server.MAX_FILES + 3)]
        result = asyncio.run(server.api_files_add({'paths': paths}))
        self.assertEqual(len(result['added']), server.MAX_FILES)
        self.assertEqual(result['limited'], 3)
        self.assertEqual(len(server.FILES), server.MAX_FILES)

    def test_duplicate_paths_do_not_consume_file_limit(self):
        first = asyncio.run(server.api_files_add({'paths': ['C:/fixtures/same.wav']}))
        repeated = asyncio.run(server.api_files_add({'paths': ['C:/fixtures/same.wav'] * 55}))
        self.assertEqual(len(first['added']), 1)
        self.assertEqual(repeated['skipped'], 55)
        self.assertEqual(repeated['limited'], 0)
        self.assertEqual(len(server.FILES), 1)

    def test_drop_rejects_before_upload_when_file_queue_is_full(self):
        for index in range(server.MAX_FILES):
            server.FILES[str(index)] = {'path': f'C:/fixtures/{index}.wav'}
        request = MagicMock()
        request.headers = {}
        with self.assertRaises(server.HTTPException) as ctx:
            asyncio.run(server.api_file_drop(request, 'extra.wav'))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(server._drop_reservations, 0)

    def test_completed_task_archive_and_restore_preserve_result_reference(self):
        task = {
            'id': 'done-one', 'path': 'C:/fixtures/done.wav', 'name': 'done.wav',
            'format': 'srt', 'prompt': '', 'status': 'done', 'error': None,
            'progress': 100.0, 'partial_text': 'finished', 'warning': None,
            'output_path': 'C:/output/done.srt', 'output_name': 'done.srt',
            'duration': 3.0, 'message': '', 'cancel_requested': False,
            'model_id': 'Qwen3-ASR-0.6B-hf', 'managed_upload': False,
            'created_at': 1.0, 'started_at': 2.0, 'finished_at': 5.0,
            'archived': False, 'archived_at': None,
        }
        server.TASKS[task['id']] = task
        server.FILES['file-one'] = {
            'id': 'file-one', 'path': task['path'], 'name': task['name'],
            'ok': True, 'reason': '', 'managed_upload': False,
        }
        with patch.object(server, 'persist_now'):
            asyncio.run(server.api_task_archive(task['id']))
            state = asyncio.run(server.api_state())
            self.assertTrue(state['tasks'][0]['archived'])
            self.assertEqual(state['files'][0]['done_formats']['srt'], task['id'])
            asyncio.run(server.api_task_restore(task['id']))
        self.assertFalse(task['archived'])
        self.assertIsNone(task['archived_at'])

    def test_archive_only_accepts_completed_tasks_and_bulk_skips_archived(self):
        server.TASKS.update({
            'queued': {'id': 'queued', 'status': 'queued', 'name': 'queued.wav', 'format': 'txt'},
            'done': {'id': 'done', 'status': 'done', 'name': 'done.wav', 'format': 'txt', 'archived': False},
            'old': {'id': 'old', 'status': 'done', 'name': 'old.wav', 'format': 'txt', 'archived': True},
        })
        with patch.object(server, 'persist_now'):
            with self.assertRaises(server.HTTPException) as ctx:
                asyncio.run(server.api_task_archive('queued'))
            self.assertEqual(ctx.exception.status_code, 409)
            result = asyncio.run(server.api_tasks_archive_completed())
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['archived'], ['done'])
        self.assertTrue(server.TASKS['done']['archived'])

    def test_tls_uvicorn_options_require_and_use_certificate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            cert = Path(directory) / 'cert.pem'
            key = Path(directory) / 'key.pem'
            with patch.object(server, 'HTTPS_ENABLED', True), \
                 patch.object(server, 'SSL_CERTFILE', cert), \
                 patch.object(server, 'SSL_KEYFILE', key):
                with self.assertRaisesRegex(RuntimeError, 'HTTPS'):
                    server._uvicorn_kwargs()
                cert.write_text('certificate', encoding='ascii')
                key.write_text('private key', encoding='ascii')
                options = server._uvicorn_kwargs()
                self.assertEqual(options['ssl_certfile'], str(cert))
                self.assertEqual(options['ssl_keyfile'], str(key))

    def test_test_mode_keeps_http_for_isolated_browser_regression(self):
        self.assertFalse(server.HTTPS_ENABLED)
        self.assertEqual(server.SERVICE_URL, 'http://127.0.0.1:8001')
        self.assertNotIn('ssl_certfile', server._uvicorn_kwargs())
        with patch.object(server, '_torch_ok', return_value=False):
            health = asyncio.run(server.api_health())
        self.assertFalse(health['https'])
        self.assertEqual(health['url'], server.SERVICE_URL)
        self.assertEqual(health['pid'], os.getpid())

    def test_launcher_uses_builtin_windows_powershell(self):
        launcher = (BASE_DIR / '启动Lvats.bat').read_bytes().decode('gbk')
        self.assertNotIn('\npwsh ', launcher.replace('\r\n', '\n'))
        self.assertIn(r'%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe', launcher)

    def test_launcher_serializes_start_and_requires_cuda_health(self):
        launcher = (BASE_DIR / '启动Lvats.bat').read_bytes().decode('gbk')
        starter = (BASE_DIR / 'scripts' / 'start_lvats.ps1').read_text(encoding='utf-8')
        self.assertIn('start_lvats.ps1', launcher)
        self.assertIn('[IO.FileShare]::None', starter)
        self.assertIn('$health.cuda', starter)
        self.assertIn('torch.cuda.is_available()', starter)

    def test_restart_strictly_saves_before_helper_and_requeues_running_task(self):
        token = MagicMock()
        server.TASKS['running'] = {'status': 'running', 'cancel_requested': False, '_token': token}
        with patch.object(server, '_restart_preflight'), \
             patch.object(server, 'persist_now', return_value=True) as persist, \
             patch.object(server, '_write_restart_status'), \
             patch.object(server, '_spawn_restart_helper', return_value=4321) as spawn, \
             patch.object(server, '_schedule_process_exit') as schedule:
            result = asyncio.run(server.api_restart())
        persist.assert_called_once_with(strict=True)
        spawn.assert_called_once_with(result['token'])
        self.assertTrue(server.TASKS['running']['cancel_requested'])
        token.set.assert_called_once()
        schedule.assert_called_once()
        self.assertEqual(result['helper_pid'], 4321)
        self.assertEqual(result['interrupted'], 1)

    def test_restart_persistence_failure_keeps_service_running(self):
        with patch.object(server, '_restart_preflight'), \
             patch.object(server, 'persist_now', side_effect=RuntimeError('disk full')), \
             patch.object(server, '_write_restart_status'), \
             patch.object(server, '_spawn_restart_helper') as spawn:
            with self.assertRaises(server.HTTPException) as ctx:
                asyncio.run(server.api_restart())
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertFalse(server._shutdown_started)
        spawn.assert_not_called()

    def test_restart_helper_reuses_windows_launcher_and_test_mode_command(self):
        helper = server.RESTART_HELPER.read_text(encoding='utf-8')
        self.assertIn('start_lvats.ps1', helper)
        self.assertIn('WindowsPowerShell', helper)
        self.assertIn('state.mkdir(parents=True, exist_ok=True)', helper)
        with patch.dict(os.environ, {'ASR_RESTART_TEST_COMMAND': '["python", "serve.py"]'}), \
             patch.object(server.subprocess, 'Popen') as popen:
            popen.return_value.pid = 123
            self.assertEqual(server._spawn_restart_helper('token'), 123)
        command = popen.call_args.args[0]
        self.assertIn('--test-mode', command)
        self.assertIn('--test-command-json', command)

    def test_http_request_redirects_to_same_https_port_and_path(self):
        response = server._http_redirect_response(b'GET /api/health?fresh=1 HTTP/1.1\r\n')
        self.assertIn(b'HTTP/1.1 308 Permanent Redirect', response)
        self.assertIn((f'Location: {server.SERVICE_URL}/api/health?fresh=1\r\n').encode(), response)

    def test_pid_cleanup_only_removes_current_process_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'PID_FILE', Path(directory) / 'pid.json'):
            server.PID_FILE.write_text('{"pid": 999999}', encoding='utf-8')
            self.assertFalse(server._remove_owned_pid_file())
            self.assertTrue(server.PID_FILE.exists())
            server.PID_FILE.write_text('{"pid": %d}' % os.getpid(), encoding='utf-8')
            self.assertTrue(server._remove_owned_pid_file())
            self.assertFalse(server.PID_FILE.exists())

    def test_orphan_upload_cleanup_retains_only_active_tasks(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'UPLOAD_DIR', Path(directory)):
            keep = Path(directory) / 'keep.wav'
            orphan = Path(directory) / 'orphan.wav'
            partial = Path(directory) / '.interrupted.part'
            for path in (keep, orphan, partial):
                path.write_bytes(b'data')
            server.TASKS['keep'] = {
                'path': str(keep), 'status': 'queued', 'managed_upload': True,
            }
            server.TASKS['done'] = {
                'path': str(orphan), 'status': 'done', 'managed_upload': True,
            }
            server._cleanup_orphan_uploads()
            self.assertTrue(keep.exists())
            self.assertFalse(orphan.exists())
            self.assertFalse(partial.exists())

    def test_managed_upload_waits_for_running_task_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'UPLOAD_DIR', Path(directory)):
            uploaded = Path(directory) / 'running.wav'
            uploaded.write_bytes(b'data')
            server.TASKS['running'] = {
                'path': str(uploaded), 'status': 'running', 'managed_upload': True,
            }
            self.assertFalse(server._cleanup_managed_if_unused(uploaded, True))
            self.assertTrue(uploaded.exists())
            server.TASKS['running']['status'] = 'done'
            self.assertTrue(server._cleanup_managed_if_unused(uploaded, True))
            self.assertFalse(uploaded.exists())

    def test_partial_alignment_failure_retains_all_text(self):
        cb = MagicMock()
        with patch.object(asr, '_model', object()), \
             patch.object(asr, 'load_model'), patch.object(asr, '_align_disabled', False), \
             patch.object(asr.wav_cache, 'acquire', return_value=(Path('sample.wav'), True)), \
             patch.object(asr, '_duration', return_value=60.0), \
             patch.object(asr, 'CHUNK_SECONDS', 30), \
             patch.object(asr, '_split_wav', return_value=[Path('fake/a.wav'), Path('fake/b.wav')]), \
             patch.object(asr.shutil, 'rmtree'), \
             patch.object(asr, '_transcribe_one_chunk', side_effect=[('first', 'Chinese', False), ('second', 'Chinese', False)]), \
             patch.object(asr, '_align', side_effect=[[{'start': 0, 'end': 4, 'text': 'first'}], RuntimeError('alignment failed')]):
            result = asr._transcribe_locked(Path('sample.wav'), progress_cb=cb)
        self.assertEqual(''.join(s['text'] for s in result['segments']), 'firstsecond')
        self.assertEqual(result['segments'][-1]['start'], 60.0)
        self.assertEqual(result['segments'][-1]['end'], 120.0)
        self.assertTrue(result['warning'])

    def test_sse_resumes_from_standard_header(self):
        async def check():
            server.BUS.publish('log', text='old')
            server.BUS.publish('log', text='new')
            request = Request({'type': 'http', 'headers': [(b'last-event-id', b'1')]})
            response = await server.api_events(request)
            iterator = response.body_iterator
            await anext(iterator)
            await anext(iterator)  # stream generation handshake
            event = await anext(iterator)
            self.assertIn('id: 2\n', event)
            await iterator.aclose()
        asyncio.run(check())

    def test_unload_rejected_during_task(self):
        server.TASKS['test'] = {'status': 'running'}
        async def check():
            with self.assertRaises(server.HTTPException) as ctx:
                await server.api_unload()
            self.assertEqual(ctx.exception.status_code, 409)
        asyncio.run(check())

    def test_preload_honors_selected_engine_and_path(self):
        async def check():
            for model_id, engine in [('Qwen3-ASR-0.6B-hf', asr), ('faster-whisper-tiny', fw)]:
                with patch.object(engine, 'preload') as preload:
                    result = await server.api_preload({'model_id': model_id})
                    await server._preload_task
                    self.assertEqual(result['model_id'], model_id)
                    self.assertTrue(preload.called)
                    arg = preload.call_args.args[0]
                    self.assertEqual(arg if engine is asr else arg['path'], registry.get(model_id)['path'])
        asyncio.run(check())

    def test_fw_model_switch_releases_before_loading(self):
        fake = MagicMock()
        paths = []
        def load(path, **kwargs):
            self.assertIsNone(fw._model)
            paths.append(path)
            return object()
        fake.WhisperModel.side_effect = load
        with patch.dict(sys.modules, {'faster_whisper': fake}), \
             patch.object(asr, 'USE_PLACEHOLDER', False):
            fw._load('small', 'a')
            fw._load('small', 'a')
            fw._load('large', 'b')
        self.assertEqual(paths, ['small', 'large'])

    def test_health_reports_whisper_and_cpu_degradation(self):
        with patch.object(fw, '_model', object()), patch.object(fw, '_model_id', 'tiny'), \
             patch.object(fw, '_device', 'cpu'), patch.object(fw, '_degraded', True), \
             patch.object(server, '_torch_ok', return_value=False):
            health = asyncio.run(server.api_health())
        self.assertTrue(health['model_loaded'])
        self.assertEqual(health['device_info']['engine'], 'faster-whisper')
        self.assertTrue(health['device_info']['degraded'])

    def test_qwen_health_refreshes_aligner_state(self):
        with patch.object(asr, 'USE_PLACEHOLDER', False), \
             patch.object(asr, '_device_info', {'aligner_loaded': False}), \
             patch.object(asr, '_aligner_model', object()), patch.object(asr, '_aligner_quantized', True):
            info = asr.get_device_info()
            self.assertTrue(info['aligner_loaded'])
            self.assertTrue(info['aligner_quantized'])

    def test_concurrent_events_have_ordered_unique_sequence(self):
        bus = server.EventBus(history=1000)
        threads = [threading.Thread(target=lambda: [bus.publish('log') for _ in range(100)]) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertEqual([event['seq'] for event in bus.since(0)], list(range(1, 501)))

    def test_idle_unloads_both_engines_but_never_interrupts_inference(self):
        ready, release = threading.Event(), threading.Event()
        def hold_inference():
            with asr.INFERENCE_LOCK:
                ready.set()
                release.wait(5)
        holder = threading.Thread(target=hold_inference)
        with patch.object(asr, 'is_loaded', return_value=True), \
             patch.object(fw, 'is_loaded', return_value=True), \
             patch.object(asr, 'seconds_since_last_use', return_value=99999), \
             patch.object(fw, 'seconds_since_last_use', return_value=99999), \
             patch.object(asr, 'unload_model') as q_unload, patch.object(fw, 'unload') as f_unload:
            holder.start()
            self.assertTrue(ready.wait(2))
            try:
                server._unload_idle_models()
                q_unload.assert_not_called()
                f_unload.assert_not_called()
            finally:
                release.set()
                holder.join(2)
            server._unload_idle_models()
            q_unload.assert_called_once()
            f_unload.assert_called_once()

    def test_sse_replay_deduplicates_and_recovers_after_restart(self):
        async def check():
            one = server.BUS.publish('log', text='one')
            request = Request({'type': 'http', 'headers': []})
            request.is_disconnected = AsyncMock(return_value=False)
            response = await server.api_events(request, last_event_id='100', stream_id='old-process')
            iterator = response.body_iterator
            await anext(iterator)
            await anext(iterator)
            self.assertIn('id: 1\n', await anext(iterator))
            two = server.BUS.publish('log', text='two')
            server.BUS._fanout(one)
            server.BUS._fanout(two)
            self.assertIn('id: 2\n', await asyncio.wait_for(anext(iterator), 1))
            await iterator.aclose()
            self.assertEqual(len(server.BUS._subs), 0)
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
