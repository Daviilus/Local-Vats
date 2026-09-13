"""Isolated HTTPS gateway smoke: plaintext redirect and TLS on one public port."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time
import uuid

BASE = Path(__file__).resolve().parents[1]
FRONT_PORT = 8100
BACKEND_PORT = 8101


def serve(run: Path):
    sys.path.insert(0, str(BASE))
    import server
    import uvicorn
    server.STATE_DIR = run / 'state'
    server.CONFIG_DIR = run / 'config'
    server.OUTPUT_DIR = run / 'output'
    server.UPLOAD_DIR = run / 'dropcache'
    server.QUEUE_FILE = server.STATE_DIR / 'queue.test.json'
    server.PID_FILE = server.STATE_DIR / 'lvats.test.pid'
    server.QUICK_PROMPTS_FILE = server.CONFIG_DIR / 'quick_prompts.test.json'
    server.model_registry.REGISTRY_FILE = server.STATE_DIR / 'registry.test.json'
    asyncio.run(server._serve_https_with_redirect(uvicorn))


def main():
    import httpx
    for port in (FRONT_PORT, BACKEND_PORT):
        with socket.socket() as probe:
            assert probe.connect_ex(('127.0.0.1', port)) != 0, f'{port} already occupied'
    run = BASE / '.cowork-temp' / ('https-redirect-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    root_der = (BASE / 'config/ssl/lvats-local-root.cer').read_bytes()
    root_pem = run / 'lvats-local-root.pem'
    root_pem.write_text(ssl.DER_cert_to_PEM_cert(root_der), encoding='ascii')
    context = ssl.create_default_context(cafile=str(root_pem))
    env = os.environ.copy()
    env.update(ASR_TEST_MODE='1', ASR_PLACEHOLDER='1', ASR_PORT=str(FRONT_PORT),
               ASR_HTTPS='1', ASR_HTTP_REDIRECT='1', ASR_TLS_BACKEND_PORT=str(BACKEND_PORT),
               ASR_SSL_CERTFILE=str(BASE / 'config/ssl/lvats-cert.pem'),
               ASR_SSL_KEYFILE=str(BASE / 'config/ssl/lvats-key.pem'))
    protected = [BASE / p for p in ['state/queue.json', 'state/registry.json',
                 'state/notified.json', 'config/quick_prompts.json']]
    def fingerprint(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    hashes = {str(path): fingerprint(path) for path in protected}
    log = (run / 'server.log').open('w', encoding='utf-8')
    process = subprocess.Popen([sys.executable, __file__, '--serve', str(run)], cwd=BASE,
                               env=env, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    https_url = f'https://127.0.0.1:{FRONT_PORT}'
    client = httpx.Client(verify=context, trust_env=False)
    try:
        deadline = time.monotonic() + 25
        health = None
        last_error = None
        while time.monotonic() < deadline:
            try:
                response = client.get(https_url + '/api/health', timeout=2)
                if response.status_code == 200:
                    health = response.json()
                    break
            except httpx.HTTPError as exc:
                last_error = repr(exc)
                time.sleep(0.15)
        assert health and health['version'] == '1.0.0' and health['https'] and health['http_redirect'], {
            'health': health, 'last_error': last_error,
        }
        redirect = client.get(f'http://127.0.0.1:{FRONT_PORT}/api/health?fresh=1',
                              follow_redirects=False, timeout=3)
        assert redirect.status_code == 308
        assert redirect.headers['location'] == https_url + '/api/health?fresh=1'
        assert client.get(redirect.headers['location'], timeout=3).status_code == 200
        client.post(https_url + '/api/shutdown', timeout=3)
        process.wait(timeout=12)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        client.close()
        log.close()
    assert hashes == {str(path): fingerprint(path) for path in protected}
    report = {'passed': ['HTTP 308 preserves path/query', 'HTTPS works on the same public port',
                         'production state files unchanged'], 'artifacts': str(run)}
    (run / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--serve':
        serve(Path(sys.argv[2]))
    else:
        main()
