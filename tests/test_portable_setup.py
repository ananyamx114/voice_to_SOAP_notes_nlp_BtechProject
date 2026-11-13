from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.check_system import physical_memory
from scripts.project import command_for, environment_python


class PortableSetupTests(unittest.TestCase):
    def test_platform_interpreters_and_argument_boundaries(self):
        root = Path("a project with spaces")
        self.assertEqual(environment_python(root, True), root / ".venv/Scripts/python.exe")
        self.assertEqual(environment_python(root, False), root / ".venv/bin/python")
        command = command_for("cli", root, ["transcribe-audio", "--audio", "a recording.wav"], True)
        self.assertEqual(command[-1], "a recording.wav")
        self.assertEqual(command[1:3], ["-m", "voice_to_soap.cli"])
        self.assertTrue(command_for("hf-login", root, [], True)[0].endswith("hf.exe"))

    def test_linux_memory_uses_page_size_and_count(self):
        with patch("scripts.check_system.os.sysconf", side_effect=[1024, 4096], create=True):
            self.assertEqual(physical_memory("Linux"), 1024 * 4096)

    def test_macos_memory_queries_only_memory_size(self):
        with patch("scripts.check_system.subprocess.run", return_value=SimpleNamespace(stdout="8589934592\n")) as run:
            self.assertEqual(physical_memory("Darwin"), 8589934592)
            self.assertEqual(run.call_args.args[0], ["sysctl", "-n", "hw.memsize"])

    def test_missing_memory_access_does_not_invent_a_value(self):
        with patch("scripts.check_system.os.sysconf", side_effect=OSError, create=True):
            self.assertIsNone(physical_memory("Linux"))

    def test_windows_memory_uses_system_api(self):
        def fill_status(pointer):
            pointer._obj.physical = 16 * 1024**3
            return 1
        api = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=fill_status))
        with patch("scripts.check_system.ctypes.windll", api, create=True):
            self.assertEqual(physical_memory("Windows"), 16 * 1024**3)
