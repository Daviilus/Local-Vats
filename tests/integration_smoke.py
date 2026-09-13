"""8001-only placeholder API/browser smoke; writes only a unique test directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import wave

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
URL = 'http://127.0.0.1:8001'


def prepare_models(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for name in ('Qwen3-ASR-0.6B-hf', 'Qwen3-ASR-1.7B-hf'):
        model = root / name
        model.mkdir(exist_ok=True)
        (model / 'config.json').write_text(json.dumps({
            'architectures': ['Qwen3ASRForConditionalGeneration'],
        }), encoding='utf-8')
        (model / 'model.safetensors').touch()
    whisper = root / 'faster-whisper-tiny'
    whisper.mkdir(exist_ok=True)
    (whisper / 'model.bin').touch()
    (whisper / 'tokenizer.json').write_text('{}', encoding='utf-8')


def serve(directory):
    os.environ.update(ASR_TEST_MODE='1', ASR_PLACEHOLDER='1', ASR_PORT='8001',
                      ASR_PLACEHOLDER_SLOW='1', ASR_WAV_CACHE_DIR=str(directory / 'cache'),
                      ASR_RESTART_TEST_COMMAND=json.dumps(
                          [sys.executable, __file__, '--serve', str(directory)]))
    import server
    import uvicorn
    server.STATE_DIR = directory / 'state'
    server.CONFIG_DIR = directory / 'config'
    server.OUTPUT_DIR = directory / 'output'
    server.UPLOAD_DIR = directory / 'dropcache'
    server.QUEUE_FILE = server.STATE_DIR / 'queue.test.json'
    server.PID_FILE = server.STATE_DIR / 'lvats.test.pid'
    server.RESTART_STATUS_FILE = server.STATE_DIR / 'restart.test.json'
    server.QUICK_PROMPTS_FILE = server.CONFIG_DIR / 'quick_prompts.test.json'
    server.model_registry.REGISTRY_FILE = server.STATE_DIR / 'registry.test.json'
    server.model_registry.MODELS_DIR = directory / 'models'
    server.model_registry.QWEN_DIR_CANDIDATES = []
    uvicorn.run(server.app, host='127.0.0.1', port=8001, log_level='warning')


def main():
    import httpx
    from playwright.sync_api import sync_playwright
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', 8001)) != 0, '8001 already occupied'
    run = BASE / '.cowork-temp' / ('v100-smoke-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    prepare_models(run / 'models')
    protected = [BASE / p for p in ['state/queue.json', 'state/registry.json',
                 'state/notified.json', 'config/quick_prompts.json']]
    def fingerprint(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    hashes = {str(p): fingerprint(p) for p in protected}
    fixture = run / 'sample.wav'
    with wave.open(str(fixture), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\0\0' * 16000 * 5)
    client = httpx.Client(base_url=URL, timeout=5, trust_env=False)
    process = None
    log = (run / 'server.log').open('w', encoding='utf-8')
    checks = []

    def api(path, data=None, method=None):
        response = client.request(method or ('POST' if data is not None else 'GET'), path, json=data)
        response.raise_for_status()
        return response.json()

    def wait_for(predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = predicate()
                if result:
                    return result
            except (httpx.ConnectError, httpx.ReadError):
                pass
            time.sleep(0.15)
        raise AssertionError('condition timed out')

    def start():
        nonlocal process
        process = subprocess.Popen([sys.executable, __file__, '--serve', str(run)],
                    cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        health = wait_for(lambda: api('/api/health'))
        assert health['placeholder'] and health['test_mode']
        assert health['version'] == '1.0.0'

    def stop():
        try:
            api('/api/shutdown', {})
        except Exception:
            pass
        if process and process.poll() is None:
            process.wait(timeout=12)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', 8001)) != 0:
                    return
            time.sleep(0.15)
        raise AssertionError('isolated port 8001 was not released')

    try:
        start()
        model_payload = api('/api/models')
        assert [item['repo_id'] for item in model_payload['builtin']] == [
            'Qwen/Qwen3-ASR-0.6B-hf', 'Qwen/Qwen3-ASR-1.7B-hf'
        ]
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True)
            page = browser.new_page(viewport={'width': 1400, 'height': 1100})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.add_init_script('''
                window.testSources = [];
                const Native = window.EventSource;
                window.EventSource = class extends Native {
                    constructor(url) { super(url); window.testSources.push(this); }
                };
                window.showSaveFilePicker = undefined;
            ''')
            page.goto(URL, wait_until='domcontentloaded')
            page.wait_for_function("document.querySelector('#modelSelect').value === 'Qwen3-ASR-0.6B-hf'")
            assert page.locator('#modelSelect optgroup').count() == 3
            assert page.locator('#dlPreset option').evaluate_all(
                "options => options.slice(1).map(o => o.value)"
            ) == ['Qwen/Qwen3-ASR-0.6B-hf', 'Qwen/Qwen3-ASR-1.7B-hf']
            checks.append('default 0.6B, three model groups, and Qwen-only built-in download list')
            prompt_height = page.locator('#promptInput').bounding_box()['height']
            assert abs(prompt_height - 76) < 1, prompt_height
            checks.append('prompt input height doubled from 38px to 76px')
            layout = page.evaluate('''() => {
                const rect = selector => document.querySelector(selector).getBoundingClientRect();
                const queue = rect('.file-queue-card'), models = rect('.model-card');
                const load = rect('.file-load-card'), output = rect('.output-card');
                const current = rect('.current-task-card');
                const consoleBox = rect('.console');
                const monitor = rect('.monitor-grid'), tasks = rect('.task-queue-card');
                return {
                    threeColumnsInOrder: queue.right < models.left && models.right < load.left,
                    columnsAligned: Math.abs(queue.top - models.top) < 2 && Math.abs(models.top - load.top) < 2
                        && Math.abs(queue.bottom - output.bottom) < 2 && Math.abs(models.bottom - output.bottom) < 2,
                    monitorsSideBySide: current.right < consoleBox.left && Math.abs(current.top - consoleBox.top) < 2,
                    monitorsAboveTaskQueue: monitor.bottom < tasks.top,
                };
            }''')
            assert all(layout.values()), layout
            checks.append('three-column workspace, pinned monitors, and lower task queue layout')
            assert page.locator('.workspace-grid > .file-queue-card').count() == 1
            assert page.locator('.workspace-grid > .model-card').count() == 1
            assert page.locator('.workspace-grid > .file-load-card').count() == 1
            assert page.locator('.workspace-grid > .output-card').count() == 1
            assert page.locator('.monitor-grid > .current-task-card').count() == 1
            assert page.locator('.monitor-grid > .console').count() == 1
            checks.append('queue-model-tools workspace and side-by-side current-task/console layout')
            page.locator('#fileCollapseBtn').click()
            assert page.locator('#fileQueueBody').is_hidden()
            page.locator('#fileCollapseBtn').click()
            page.locator('#taskCollapseBtn').click()
            assert page.locator('#taskQueueBody').is_hidden()
            page.locator('#taskCollapseBtn').click()
            page.locator('#archiveCollapseBtn').click()
            assert page.locator('#archiveBody').is_hidden()
            assert page.evaluate("localStorage.getItem('lvats.archiveCollapsed')") == '1'
            page.reload(wait_until='domcontentloaded')
            page.wait_for_function("document.querySelector('#modelSelect').value === 'Qwen3-ASR-0.6B-hf'")
            assert page.locator('#archiveBody').is_hidden()
            assert page.locator('#archiveCollapseBtn').get_attribute('aria-expanded') == 'false'
            page.locator('#archiveCollapseBtn').click()
            assert page.locator('#archiveBody').is_visible()
            checks.append('file, task, and archived-task collapse controls with persisted preference')
            page.evaluate('''() => {
                const transfer = new DataTransfer();
                transfer.items.add(new File([new Uint8Array([73, 68, 51, 4])], 'dragged.mp3', {type: 'audio/mpeg'}));
                const zone = document.querySelector('#fileDropZone');
                zone.dispatchEvent(new DragEvent('dragenter', {bubbles: true, cancelable: true, dataTransfer: transfer}));
                zone.dispatchEvent(new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: transfer}));
            }''')
            dropped = wait_for(lambda: next((f for f in api('/api/state')['files'] if f['name'] == 'dragged.mp3'), None))
            dropped_path = Path(dropped['path'])
            assert dropped['managed_upload'] and dropped_path.is_file()
            assert dropped_path.parent == run / 'dropcache'
            page.locator('.upload-mark').wait_for()
            page.locator(f'.rm[data-fid="{dropped["id"]}"]').click()
            wait_for(lambda: not dropped_path.exists())
            checks.append('browser drag-drop upload, managed-copy badge, and removal cleanup')
            page_files = []
            for index in range(11):
                path = run / f'page-{index:02d}.wav'
                path.write_bytes(fixture.read_bytes())
                page_files.append(str(path))
            added_page_files = api('/api/files/add', {'paths': page_files})['added']
            page.wait_for_function("document.querySelectorAll('#fileList > li').length === 10")
            assert page.locator('#filePageInfo').inner_text() == '1 / 2'
            page.locator('#checkAll').check()
            page.locator('#fileNext').click()
            assert page.locator('#fileList > li').count() == 1
            assert page.locator('#fileList input[type=checkbox]').is_checked()
            assert page.locator('#filePageInfo').inner_text() == '2 / 2'
            for record in added_page_files:
                api('/api/files/remove', {'file_id': record['id']})
            page.wait_for_function("document.querySelectorAll('#fileList > li').length === 0")
            checks.append('ten-file pagination, page clamping, and cross-page select-all')
            page.locator('#pasteBox').fill(str(fixture))
            page.locator('#pasteAdd').click()
            page.locator('#fileList input[type=checkbox]').wait_for()
            for fmt, model in [('srt', 'Qwen3-ASR-0.6B-hf'), ('txt', 'faster-whisper-tiny'), ('vtt', 'Qwen3-ASR-1.7B-hf')]:
                page.locator('#modelSelect').select_option(model)
                page.locator('#fileList input[type=checkbox]').check()
                page.locator(f'.fmt-btn[data-fmt={fmt}]').click()
            page.locator('.rowbar-fill').first.wait_for(state='visible')
            assert page.locator('#asciiProgress').is_visible()
            assert page.locator('#currentTaskName').inner_text() == 'sample.wav'
            page.screenshot(path=str(run / 'progress.png'), full_page=True)
            state = api('/api/state')
            queued = [t for t in state['tasks'] if t['status'] == 'queued']
            assert len(queued) == 2
            moved = queued[-1]['id']
            api('/api/tasks/' + moved + '/move', {'delta': -1})
            assert api('/api/state')['order'][0] == moved
            assert client.post('/api/unload').status_code == 409
            checks.append('both progress bars, queue reorder, busy unload guard')

            page.wait_for_function("window.testSources.length > 0 && window.testSources[0].readyState === 1")
            page.evaluate("window.testSources[0].close(); window.testSources[0].onerror(new Event('error'))")
            page.wait_for_function("window.testSources.length > 1")
            reconnect_url = page.evaluate("window.testSources.at(-1).url")
            assert 'last_event_id=' in reconnect_url and 'stream_id=' in reconnect_url
            checks.append('browser reconnect sends cursor and stream id')

            state = wait_for(lambda: (s if len(s['tasks']) == 3 and all(t['status'] == 'done' for t in s['tasks']) else None)
                             if (s := api('/api/state')) else None)
            for t in state['tasks']:
                response = client.get('/api/read-result', params={'task_id': t['id']})
                assert response.status_code == 200 and response.content
                if t['format'] == 'srt':
                    assert '-->' in response.text
                if t['format'] == 'vtt':
                    assert response.text.startswith('WEBVTT')
            checks.append('Qwen/Whisper placeholder queue and TXT/SRT/VTT downloads')
            page.locator('.fmt-done[data-fmt=srt]').wait_for()
            with page.expect_download() as downloaded:
                page.locator('.fmt-done[data-fmt=srt]').click()
            downloaded.value.save_as(run / 'browser-download.srt')
            assert (run / 'browser-download.srt').stat().st_size > 0
            checks.append('browser save/download fallback')

            archived_id = state['tasks'][0]['id']
            api('/api/tasks/' + archived_id + '/archive', {})
            page.wait_for_function("document.querySelectorAll('#archiveList > li').length === 1")
            assert page.locator('#taskList > li').count() == 2
            assert client.get('/api/read-result', params={'task_id': archived_id}).status_code == 200
            api('/api/tasks/' + archived_id + '/restore', {})
            page.wait_for_function("document.querySelectorAll('#archiveList > li').length === 0")
            bulk = api('/api/tasks/archive-completed', {})
            assert bulk['count'] == 3
            page.wait_for_function("document.querySelectorAll('#archiveList > li').length === 3")
            assert page.locator('#taskList > li').count() == 0
            page.screenshot(path=str(run / 'archive.png'), full_page=True)
            for task_id in bulk['archived']:
                api('/api/tasks/' + task_id + '/restore', {})
            page.wait_for_function("document.querySelectorAll('#taskList > li').length === 3")
            checks.append('completed-task individual/bulk archive, download preservation, and restore')

            persisted_archived_id = bulk['archived'][0]
            api('/api/tasks/' + persisted_archived_id + '/archive', {})

            prompt = api('/api/prompts', {'name': 'smoke prompt', 'content': 'test words'})['prompt']
            api('/api/prompts/' + prompt['id'], {'content': 'changed words'}, method='PUT')
            ids_before = {t['id']: t['model_id'] for t in state['tasks']}
            stop()
            start()
            restored = api('/api/state')
            assert {t['id']: t['model_id'] for t in restored['tasks']} == ids_before
            assert next(t for t in restored['tasks'] if t['id'] == persisted_archived_id)['archived']
            api('/api/tasks/' + persisted_archived_id + '/restore', {})
            assert api('/api/prompts')['prompts'][0]['content'] == 'changed words'
            api('/api/prompts/' + prompt['id'], method='DELETE')
            checks.append('restart preserves archived state, task model ids, and prompt edits')
            page.wait_for_function("window.testSources.at(-1).readyState === 1", timeout=20000)
            # New tasks after restart confirm the SSE generation reset does not suppress events.
            api('/api/files/add', {'paths': [str(fixture)]})
            fid = api('/api/state')['files'][0]['id']
            api('/api/tasks/add', {'file_ids': [fid], 'format': 'txt', 'model_id': 'faster-whisper-tiny'})
            running = wait_for(lambda: next((t for t in api('/api/state')['tasks'] if t['status'] == 'running'), None))
            api('/api/tasks/' + running['id'] + '/cancel', {})
            wait_for(lambda: any(t['id'] == running['id'] and t['status'] == 'cancelled' for t in api('/api/state')['tasks']))
            api('/api/tasks/' + running['id'], method='DELETE')
            checks.append('cooperative cancellation and task deletion')

            old_pid = api('/api/health')['pid']
            page.on('dialog', lambda dialog: dialog.accept())
            page.locator('#restartBtn').click()
            page.wait_for_function(
                """async oldPid => {
                  try {
                    const h = await (await fetch('/api/health', {cache: 'no-store'})).json();
                    return h.pid !== oldPid && h.version === '1.0.0';
                  } catch (_) { return false; }
                }""",
                arg=old_pid, timeout=65000,
            )
            page.wait_for_function("document.querySelector('#downBanner').hidden", timeout=15000)
            assert api('/api/restart/status')['phase'] == 'ready'
            checks.append('restart button saves state, replaces the process, reports progress, and reconnects')
            page.screenshot(path=str(run / 'complete.png'), full_page=True)
            assert not errors, errors
            checks.append('no browser JavaScript errors')
            browser.close()
    finally:
        try:
            stop()
        finally:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            log.close()
            client.close()
        assert hashes == {str(p): fingerprint(p) for p in protected}, 'production file changed'
    checks.append('production queue/registry/prompts/notification files unchanged')
    report = {'passed': checks, 'artifacts': str(run)}
    (run / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', type=Path)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
    else:
        main()
