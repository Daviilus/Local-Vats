"""Exercise BAT argument parsing with the Windows PowerShell used by Explorer."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == 'nt', 'Windows launcher integration')
class WindowsLauncher(unittest.TestCase):
    def test_bat_passes_project_root_and_port(self):
        project = Path(__file__).resolve().parents[1]
        launcher = (project / '启动Lvats.bat').read_bytes().decode('gbk')
        command = next(line for line in launcher.splitlines()
                       if 'start_lvats.ps1' in line)
        starter = (project / 'scripts/start_lvats.ps1').read_text(encoding='utf-8-sig')
        # Execute the real argument declaration and path resolution, stopping
        # before state creation, health requests, or any service launch.
        prefix = starter.split("$state = Join-Path", 1)[0]
        with tempfile.TemporaryDirectory(prefix='Lvats 路径 with spaces ') as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / 'scripts/start_lvats.ps1').write_text(
                prefix + '\n@{ root = $root; port = $Port } | ConvertTo-Json | '
                'Set-Content -LiteralPath (Join-Path $root "probe.json") -Encoding UTF8\n',
                encoding='utf-8-sig')
            bat = root / 'probe.bat'
            bat.write_bytes(('@echo off\r\n' + command + '\r\n').encode('gbk'))
            result = subprocess.run(
                [os.environ['COMSPEC'], '/d', '/c', str(bat)],
                capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, repr(result.stderr))
            data = json.loads((root / 'probe.json').read_text(encoding='utf-8-sig'))
            self.assertEqual(Path(data['root']), root)
            self.assertEqual(data['port'], 8000)


if __name__ == '__main__':
    unittest.main()
