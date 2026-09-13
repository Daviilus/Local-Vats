"""Explicit real-GPU validation; refuse to run alongside either service instance.

Uses an existing short speech sample; no queue/server startup or model downloads.
Run only after the placeholder service is stopped and GPU is free.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

BASE = Path(__file__).resolve().parents[1]


def main():
    for port in (8000, 8001):
        with socket.socket() as probe:
            assert probe.connect_ex(('127.0.0.1', port)) != 0, f'port {port} is occupied; refusing real GPU test'
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert int(used.strip().splitlines()[0]) < 512, 'GPU is not idle'
    run = BASE / '.cowork-temp' / ('v071-real-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    os.environ.update(ASR_TEST_MODE='1', ASR_PLACEHOLDER='0', ASR_PORT='8001',
                      ASR_WAV_CACHE_DIR=str(run / 'cache'),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(BASE))
    import asr_model as asr
    import server
    import torch
    sample = BASE / '.cowork-temp' / 'asr_zh.wav'
    duration = asr._duration(sample)
    protected = [BASE / p for p in ['state/queue.json', 'state/registry.json',
                 'state/notified.json', 'config/quick_prompts.json']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    results = []
    print('Artifacts: ' + str(run), flush=True)
    try:
        for model, directory in [('0.6B', BASE / 'models/Qwen3-ASR-0.6B-hf'),
                                 ('1.7B', BASE / 'Qwen3-ASR-1.7B-hf')]:
            started = time.perf_counter()
            def cb(pct, message, partial=None, meta=None):
                print(f'[{model} {pct:5.1f}%] {message}', flush=True)
                if meta and meta.get('log'):
                    print(meta['log'], flush=True)
            result = asr.transcribe_audio(sample, progress_cb=cb, job_id='real-' + model,
                                          need_timestamps=True, model_dir=directory)
            info = asr.get_device_info()
            assert info['model_path'] == asr._norm_path(directory.resolve())
            assert not result.get('warning'), result.get('warning')
            assert result['text'].strip() and result['segments']
            assert not info.get('degraded')
            for segment in result['segments']:
                assert 0 <= segment['start'] < segment['end'] <= duration + 1, segment
            (run / f'qwen-{model}.srt').write_text(server.render_srt(result), encoding='utf-8')
            record = {'model': model, 'seconds': round(time.perf_counter() - started, 2),
                      'duration': duration, 'segments': len(result['segments']),
                      'device': info, 'text': result['text']}
            results.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            asr.wav_cache.release(sample, 'real-' + model)
    finally:
        asr.unload_model()
        asr.wav_cache.drop_all()
        assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        (run / 'report.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: both actual Qwen models produced aligned SRT; production files unchanged', flush=True)


if __name__ == '__main__':
    main()
