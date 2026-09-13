"""Interactive Windows regression for the native file and folder pickers.

Run explicitly with ASR_RUN_INTERACTIVE_DIALOG_TEST=1.  Each dialog is
inspected through Win32 and closed automatically; no user input is required.
"""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest


os.environ.setdefault("ASR_TEST_MODE", "1")
os.environ.setdefault("ASR_PLACEHOLDER", "1")
os.environ.setdefault("ASR_PORT", "8001")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


RUN_INTERACTIVE = os.name == "nt" and os.environ.get(
    "ASR_RUN_INTERACTIVE_DIALOG_TEST"
) == "1"


@unittest.skipUnless(RUN_INTERACTIVE, "opt-in interactive Windows dialog test")
class WindowsDialogTopmost(unittest.TestCase):
    WS_EX_TOPMOST = 0x00000008
    GWL_EXSTYLE = -20
    WM_CLOSE = 0x0010

    def _find_dialog(self, pid: int):
        user32 = ctypes.windll.user32
        candidates = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def inspect(hwnd, _):
            owner_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
            if owner_pid.value != pid or not user32.IsWindowVisible(hwnd):
                return True
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
                if area > 10_000:  # Excludes the hidden 1x1 owner form.
                    candidates.append((area, hwnd))
            return True

        user32.EnumWindows(callback_type(inspect), 0)
        return max(candidates, default=(0, None))[1]

    @staticmethod
    def _window_description(hwnd):
        user32 = ctypes.windll.user32
        title = ctypes.create_unicode_buffer(512)
        class_name = ctypes.create_unicode_buffer(256)
        owner_pid = wintypes.DWORD()
        user32.GetWindowTextW(hwnd, title, len(title))
        user32.GetClassNameW(hwnd, class_name, len(class_name))
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
        return f"pid={owner_pid.value}, class={class_name.value!r}, title={title.value!r}"

    def _assert_picker_topmost(self, script: str):
        process = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        hwnd = None
        stdout = stderr = ""
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and process.poll() is None:
                hwnd = self._find_dialog(process.pid)
                if hwnd:
                    break
                time.sleep(0.1)
            if hwnd is None and process.poll() is not None:
                stdout, stderr = process.communicate(timeout=2)
            self.assertIsNotNone(
                hwnd,
                f"native picker window did not appear; stdout={stdout!r}; stderr={stderr!r}",
            )
            user32 = ctypes.windll.user32
            get_window_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            get_window_long.restype = ctypes.c_ssize_t
            style = 0
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                style = get_window_long(hwnd, self.GWL_EXSTYLE)
                if style & self.WS_EX_TOPMOST:
                    break
                time.sleep(0.05)
            diagnostics = ""
            if not style & self.WS_EX_TOPMOST:
                user32.PostMessageW(hwnd, self.WM_CLOSE, 0, 0)
                stdout, stderr = process.communicate(timeout=10)
                diagnostics = f"; stdout={stdout!r}; stderr={stderr!r}"
            self.assertTrue(style & self.WS_EX_TOPMOST,
                            f"picker HWND {hwnd} is missing WS_EX_TOPMOST; "
                            + self._window_description(hwnd) + diagnostics)
            user32.PostMessageW(hwnd, self.WM_CLOSE, 0, 0)
            process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

    def test_file_and_folder_pickers_are_topmost(self):
        for kind, script in (("files", server.PICK_PS1),
                             ("folder", server.FOLDER_PS1)):
            with self.subTest(kind=kind):
                self._assert_picker_topmost(script)


if __name__ == "__main__":
    unittest.main()
