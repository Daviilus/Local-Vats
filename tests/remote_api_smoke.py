"""Opt-in isolated API/browser/proxy smoke with fake keys and loopback servers.

No production requests, model loading, certificate changes, or paid service calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

BASE = Path(__file__).resolve().parents[1]
TEST_PORT = int(os.environ.get('ASR_SMOKE_PORT', '8001'))
if not 1024 <= TEST_PORT <= 65535 or TEST_PORT == 8000:
    raise ValueError('ASR_SMOKE_PORT must be a non-production port from 1024 to 65535')
URL = f'http://127.0.0.1:{TEST_PORT}'
FAKE_KEY = 'smoke-only-not-a-real-secret'


class MockTranscription(BaseHTTPRequestHandler):
    requests_seen = []
    next_status = 200
    timestamps = True

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        if self.headers.get('Authorization') != 'Bearer ' + FAKE_KEY:
            self.send_response(401)
            self.end_headers()
            return
        if not self.path.endswith('/audio/transcriptions'):
            self.send_response(404)
            self.end_headers()
            return
        assert b'filename="audio.mp3"' in body and b'Content-Type: audio/mpeg' in body
        assert b'private-fixture' not in body, 'Source filename leaked to provider'
        assert b'RIFF' not in body[:200], 'Source WAV should have been converted'
        self.requests_seen.append({'path': self.path, 'bytes': len(body)})
        status = self.next_status
        type(self).next_status = 200
        payload = {'text': '模拟接口转录成功。'}
        if self.timestamps:
            payload['segments'] = [{'start': 0, 'end': 0.75, 'text': payload['text']}]
        if status != 200:
            payload = {'error': {'message': 'Unsafe upstream detail ' + FAKE_KEY}}
        encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ForwardProxy(BaseHTTPRequestHandler):
    requests_seen = []
    target_port = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        import http.client
        target = urlsplit(self.path)
        assert target.scheme == 'http' and target.hostname == '127.0.0.1'
        assert target.port == self.target_port, 'Proxy must never reach any other destination'
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        self.requests_seen.append(self.path)
        connection = http.client.HTTPConnection('127.0.0.1', self.target_port, timeout=5)
        try:
            connection.request('POST', target.path, body=body,
                               headers={'Content-Type': self.headers['Content-Type'],
                                        'Authorization': self.headers['Authorization'],
                                        'Content-Length': str(len(body))})
            response = connection.getresponse()
            data = response.read()
            self.send_response(response.status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            connection.close()


def serve(run: Path):
    os.environ.update(ASR_TEST_MODE='1', ASR_PLACEHOLDER='1', ASR_PORT=str(TEST_PORT),
                      ASR_HTTPS='0', ASR_DATA_ROOT=str(run / 'user-data'))
    sys.path.insert(0, str(BASE))
    import server
    server.serve()


def main():
    import httpx
    from playwright.sync_api import sync_playwright
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', TEST_PORT)) != 0, f'{TEST_PORT} already occupied; refusing to replace service'
    run = BASE / '.cowork-temp' / ('remote-api-smoke-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    fixture = run / 'private-fixture 中文.wav'
    with wave.open(str(fixture), 'wb') as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(16000)
        sound.writeframes(b'\0\0' * 16000)
    production_fingerprint = {}
    # Read only app identity, not production queue/media contents or credentials.
    with httpx.Client(trust_env=False, timeout=2) as plain:
        try:
            response = plain.get('http://127.0.0.1:8000/')
            production_fingerprint['status'] = response.status_code
            production_fingerprint['location'] = response.headers.get('location')
        except httpx.HTTPError:
            production_fingerprint['status'] = None
    mock = ThreadingHTTPServer(('127.0.0.1', 0), MockTranscription)
    ForwardProxy.target_port = mock.server_port
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), ForwardProxy)
    for instance in (mock, proxy):
        threading.Thread(target=instance.serve_forever, daemon=True).start()
    process = None
    log = (run / 'web.log').open('w', encoding='utf-8')
    client = httpx.Client(base_url=URL, timeout=5, trust_env=False,
                         headers={'X-Lvats-Request': '1', 'Origin': URL})
    checks = []
    errors = []
    headers = {'X-Lvats-Request': '1', 'Origin': URL}

    def api(path, payload=None, method=None):
        if path == '/api/remote/config' and payload is not None:
            payload = {**payload, 'expected_revision': client.get(path).json()['revision']}
        response = client.request(method or ('POST' if payload is not None else 'GET'), path, json=payload)
        response.raise_for_status()
        return response.json()

    def wait_for(predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                value = predicate()
                if value:
                    return value
            except (httpx.HTTPError, KeyError):
                pass
            time.sleep(0.1)
        raise AssertionError('Isolated condition timed out')

    def make_task(fmt, expected='done'):
        before = {task['id'] for task in api('/api/state')['tasks']}
        result = api('/api/tasks/add', {'file_ids': [file_id], 'format': fmt,
                                      'model_id': 'third-party-api', 'remote_consent': True,
                                      'remote_revision': api('/api/remote/config')['revision']})
        assert result['created'] == 1
        task = wait_for(lambda: next((item for item in api('/api/state')['tasks']
                                      if item['id'] not in before and item['status'] in ('done', 'failed')), None))
        assert task['status'] == expected, task.get('error')
        assert FAKE_KEY not in json.dumps(task)
        return task

    try:
        process = subprocess.Popen([sys.executable, __file__, '--serve', str(run)], cwd=BASE,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        health = wait_for(lambda: api('/api/health'))
        assert health['placeholder'] and health['test_mode']
        assert Path(health['data_root']).resolve() == (run / 'user-data').resolve()
        checks.append(f'Isolated {TEST_PORT} service and test data root, no speech model loading')
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge', headless=True)
            context = browser.new_context(viewport={'width': 1400, 'height': 1000})
            page = context.new_page()
            page.on('pageerror', lambda err: errors.append(str(err)))
            page.goto(URL, wait_until='domcontentloaded')
            page.locator('#apiSettingsLink').click()
            page.wait_for_url(URL + '/api-settings')
            page.locator('#apiUrl').wait_for(state='visible')
            page.wait_for_function('() => !document.querySelector("#apiUrl").disabled && !document.querySelector("#saveConfig").disabled')
            assert page.locator('#proxyToggle').get_attribute('aria-expanded') == 'false'
            page.locator('#apiUrl').fill(f'http://127.0.0.1:{mock.server_port}/v1')
            page.locator('#apiModel').fill('whisper-1')
            page.locator('#apiKey').fill(FAKE_KEY)
            page.locator('#proxyToggle').click()
            assert page.locator('#proxyToggle').get_attribute('aria-expanded') == 'true'
            page.locator('#proxyEnabled').check()
            page.locator('#proxyUrl').fill('http://127.0.0.1')
            page.locator('#proxyPort').fill(str(proxy.server_port))
            page.locator('#saveConfig').click()
            wait_for(lambda: api('/api/remote/config')['ready'])
            page.wait_for_function('() => document.querySelector("#apiKey").value === ""')
            assert MockTranscription.requests_seen == [], 'Save must not contact paid endpoint'
            assert ForwardProxy.requests_seen == [], 'Save must not contact proxy'
            all_storage = page.evaluate('JSON.stringify({local: {...localStorage}, session: {...sessionStorage}})')
            assert FAKE_KEY not in all_storage
            config_response = client.get('/api/remote/config')
            assert FAKE_KEY not in config_response.text
            assert config_response.headers['cache-control'] == 'no-store'
            stored = (run / 'user-data/config/remote-api.test.json').read_text(encoding='utf-8')
            assert FAKE_KEY not in stored and 'dpapi-current-user-v1' in stored
            checks.append('Separate config page, proxy expansion/apply, current-user DPAPI, no key echo/storage')
            page.reload(wait_until='domcontentloaded')
            page.wait_for_function('() => !document.querySelector("#saveConfig").disabled')
            assert page.locator('#apiKey').input_value() == ''
            page.set_viewport_size({'width': 375, 'height': 812})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile horizontal overflow'
            page.screenshot(path=str(run / 'api-settings-mobile.png'), full_page=True)
            page.set_viewport_size({'width': 1400, 'height': 1000})
            page.screenshot(path=str(run / 'api-settings-desktop.png'), full_page=True)
            page.locator('a[href="/"]').first.click()
            page.wait_for_url(URL + '/')
            page.locator('#modelSelect option[value="third-party-api"]').wait_for(state='attached')
            page.locator('#modelSelect').select_option('third-party-api')
            assert page.locator('#remoteTaskHint').is_visible()
            file_id = api('/api/files/add', {'paths': [str(fixture)]})['added'][0]['id']
            wait_for(lambda: page.locator(f'input[data-fid="{file_id}"]').count())
            page.locator(f'input[data-fid="{file_id}"]').check()
            rejected = []
            def reject(dialog):
                rejected.append(dialog.message)
                dialog.dismiss()
            page.on('dialog', reject)
            page.locator('.fmt-btn[data-fmt="txt"]').click()
            assert rejected and not api('/api/state')['tasks'], 'Rejecting consent should not create task'
            page.remove_listener('dialog', reject)
            page.on('dialog', lambda dialog: dialog.accept())
            page.locator('.fmt-btn[data-fmt="txt"]').click()
            txt = wait_for(lambda: next((task for task in api('/api/state')['tasks'] if task['status'] == 'done'), None))
            assert txt['model_label'] == 'API · whisper-1'
            assert client.get('/api/read-result', params={'task_id': txt['id']}).text.strip() == '模拟接口转录成功。'
            assert len(ForwardProxy.requests_seen) == 1 and len(MockTranscription.requests_seen) == 1
            checks.append('Explicit browser consent and actual HTTP proxy forwarding with converted audio')
            for fmt in ('srt', 'vtt'):
                task = make_task(fmt)
                text = client.get('/api/read-result', params={'task_id': task['id']}).text
                assert '00:00:00' in text and '模拟接口转录成功。' in text
                if fmt == 'vtt':
                    assert text.startswith('WEBVTT')
            checks.append('TXT/SRT/VTT task execution and valid segment timestamps through proxy')
            proxied = len(ForwardProxy.requests_seen)
            api('/api/remote/config', {'proxy': {'enabled': False}}, method='PUT')
            make_task('txt')
            assert len(ForwardProxy.requests_seen) == proxied, 'Disabling proxy must use direct connection'
            checks.append('Disabled proxy applies direct route without environment proxy')
            MockTranscription.next_status = 401
            failed = make_task('txt', 'failed')
            assert '认证失败' in failed['error'] and FAKE_KEY not in failed['error']
            MockTranscription.timestamps = False
            failed = make_task('srt', 'failed')
            assert '字幕' in failed['error']
            MockTranscription.timestamps = True
            with socket.socket() as port_probe:
                port_probe.bind(('127.0.0.1', 0))
                unavailable_port = port_probe.getsockname()[1]
            api('/api/remote/config', {'proxy': {'enabled': True, 'url': 'http://127.0.0.1', 'port': unavailable_port}}, method='PUT')
            failed = make_task('txt', 'failed')
            assert '代理' in failed['error']
            checks.append('Authentication, subtitle-unsupported and unreachable-proxy fail safely without paid retry')
            before = len(MockTranscription.requests_seen)
            invalid = client.put('/api/remote/config', json={'expected_revision': api('/api/remote/config')['revision'],
                                                             'proxy': {'enabled': True, 'url': 'http://127.0.0.1', 'port': 0}})
            assert invalid.status_code == 400 and FAKE_KEY not in invalid.text
            denied = client.put('/api/remote/config', headers={'Origin': 'https://unrelated.example'}, json={'clear_api_key': True})
            assert denied.status_code == 403
            assert len(MockTranscription.requests_seen) == before
            api('/api/remote/config', {'clear_api_key': True}, method='PUT')
            assert not api('/api/remote/config')['ready']
            assert 'third-party-api' not in [item['id'] for item in api('/api/models')['models']]
            checks.append('Invalid config and cross-origin requests rejected, explicit key clear disables remote option')
            assert errors == [], 'Browser page errors occurred'
            browser.close()
    finally:
        if process and process.poll() is None:
            try:
                api('/api/shutdown', {})
            except httpx.HTTPError:
                pass
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        client.close()
        log.close()
        for instance in (proxy, mock):
            instance.shutdown()
            instance.server_close()
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', TEST_PORT)) != 0, 'Test port still occupied'
    assert FAKE_KEY not in (run / 'web.log').read_text(encoding='utf-8')
    assert FAKE_KEY not in (run / 'user-data/data/state/queue.test.json').read_text(encoding='utf-8')
    assert list((run / 'user-data/temp').glob('remote-api-*')) == [], 'Audio temp directories leaked'
    after = {}
    with httpx.Client(trust_env=False, timeout=2) as plain:
        try:
            response = plain.get('http://127.0.0.1:8000/')
            after = {'status': response.status_code, 'location': response.headers.get('location')}
        except httpx.HTTPError:
            after = {'status': None}
    assert after == production_fingerprint, 'Production HTTP identity changed unexpectedly'
    checks.append('Test service/ports/temp cleanup, logs and queue secret-free, production HTTP identity unchanged')
    report = {'passed': checks, 'browser_errors': len(errors), 'mock_requests': len(MockTranscription.requests_seen),
              'proxy_requests': len(ForwardProxy.requests_seen), 'artifacts': str(run)}
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
