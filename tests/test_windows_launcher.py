"""Exercise BAT argument parsing with the Windows PowerShell used by Explorer."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == 'nt', 'Windows launcher integration')
class WindowsLauncher(unittest.TestCase):
    def test_start_script_encoding_and_powershell51_parse(self):
        project = Path(__file__).resolve().parents[1]
        script = project / 'scripts/start_lvats.ps1'
        self.assertTrue(script.read_bytes().startswith(b'\xef\xbb\xbf'),
                        'Windows PowerShell 5.1 requires UTF-8 BOM for this Chinese script')
        powershell = str(Path(os.environ.get('SystemRoot', r'C:\Windows')) /
                         'System32/WindowsPowerShell/v1.0/powershell.exe')
        escaped = str(script).replace("'", "''")
        command = ("$t=$null; $e=$null; "
                   "[System.Management.Automation.Language.Parser]::ParseFile("
                   f"'{escaped}',[ref]$t,[ref]$e)>$null; "
                   "$e | ForEach-Object { Write-Output $_.Message }; "
                   "if($e.Count){exit 1}")
        result = subprocess.run([powershell, '-NoProfile', '-Command', command],
                                capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, repr(result.stdout + result.stderr))

    def test_path_probe_with_isolated_python(self):
        import sys
        project = Path(__file__).resolve().parents[1]
        scratch = project / '.cowork-temp'
        scratch.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='launcher 路径 ', dir=scratch))
        starter = (project / 'scripts/start_lvats.ps1').read_text(encoding='utf-8-sig')
        prefix = starter.split('$executable = Resolve-LvatsExecutable', 1)[0]
        probe = root / 'probe.ps1'
        probe.write_text(prefix + '\n[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)\nGet-LvatsPaths $env:PROBE_PYTHON | ConvertTo-Json\n',
                         encoding='utf-8-sig')
        env = dict(os.environ, PROBE_PYTHON=sys.executable,
                   ASR_DATA_ROOT=str(root / 'data'), ASR_TEST_MODE='1')
        powershell = str(Path(os.environ.get('SystemRoot', r'C:\Windows')) /
                         'System32/WindowsPowerShell/v1.0/powershell.exe')
        result = subprocess.run([powershell, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                 '-File', str(probe), '-ProjectRoot', str(project)],
                                cwd=root, env=env, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, repr(result.stdout + result.stderr))
        data = json.loads(result.stdout)
        self.assertEqual(Path(data['DATA_ROOT']), root / 'data')
        self.assertEqual(Path(data['APP_ROOT']), project)

    def test_bat_passes_project_root_and_port(self):
        project = Path(__file__).resolve().parents[1]
        launcher = (project / '启动Lvats.bat').read_bytes().decode('gbk')
        command = next(line for line in launcher.splitlines()
                       if 'start_lvats.ps1' in line)
        starter = (project / 'scripts/start_lvats.ps1').read_text(encoding='utf-8-sig')
        # Execute the real argument declaration and root normalization, stopping
        # before runtime discovery, directory creation, health checks, or launch.
        prefix = starter.split("function Get-LvatsHealth", 1)[0]
        with tempfile.TemporaryDirectory(prefix='Lvats 路径 with spaces ') as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / 'scripts/start_lvats.ps1').write_text(
                prefix + '\n@{ root = $root; port = $Port } | ConvertTo-Json | '
                'Set-Content -LiteralPath (Join-Path $root "probe.json") -Encoding UTF8\n',
                encoding='utf-8-sig')
            bat = root / 'probe.bat'
            bat.write_bytes(('@echo off\r\n' + command + '\r\n').encode('gbk'))
            comspec = os.environ.get('COMSPEC') or str(
                Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'cmd.exe')
            result = subprocess.run(
                [comspec, '/d', '/c', str(bat)],
                capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, repr(result.stderr))
            data = json.loads((root / 'probe.json').read_text(encoding='utf-8-sig'))
            self.assertEqual(Path(data['root']), root)
            self.assertEqual(data['port'], 8000)


if __name__ == '__main__':
    unittest.main()
