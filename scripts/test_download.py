import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("download", Path(__file__).with_name("download-model.py"))
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DownloadRecoveryTests(unittest.TestCase):
    def test_corrupt_resume_retries_from_zero_and_atomically_promotes(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "weights"
            partial = Path(folder) / "weights.part"
            partial.write_bytes(b"corrupt")
            calls = []
            def curl(*_, **__):
                calls.append(partial.exists())
                partial.write_bytes(b"corrupt complete" if len(calls) == 1 else b"correct")
            with patch.object(download.subprocess, "run", side_effect=curl):
                download.fetch("repo", "rev", "weights", dest, hashlib.sha256(b"correct").hexdigest())
            self.assertEqual(calls, [True, False])
            self.assertEqual(dest.read_bytes(), b"correct")
            self.assertFalse(partial.exists())

    def test_range_error_retries_cleanly_and_second_bad_digest_leaves_no_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "weights"
            partial = Path(folder) / "weights.part"
            partial.write_bytes(b"corrupt")
            calls = 0
            def curl(*_, **__):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise subprocess.CalledProcessError(33, "curl")
                self.assertFalse(partial.exists())
                partial.write_bytes(b"bad again")
            with patch.object(download.subprocess, "run", side_effect=curl):
                with self.assertRaisesRegex(RuntimeError, "clean retry"):
                    download.fetch("repo", "rev", "weights", dest, hashlib.sha256(b"correct").hexdigest())
            self.assertFalse(partial.exists())
            self.assertFalse(dest.exists())

    def test_bad_destination_is_quarantined_and_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "weights"
            dest.write_bytes(b"wrong")
            def curl(*_, **__):
                dest.with_suffix(".part").write_bytes(b"right")
            with patch.object(download.subprocess, "run", side_effect=curl):
                download.fetch("repo", "rev", "weights", dest, hashlib.sha256(b"right").hexdigest())
            self.assertEqual(dest.read_bytes(), b"right")
            self.assertEqual(next(Path(folder).glob("weights.corrupt-*")).read_bytes(), b"wrong")


    def test_os_lock_survives_stale_pid_text_and_is_released_after_kill(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".pull.lock").write_text("stale process")
            code = "import fcntl,sys,time; f=open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); print('locked',flush=True); time.sleep(60)"
            child = subprocess.Popen([sys.executable, "-c", code, str(root / ".pull.lock")], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                with self.assertRaisesRegex(RuntimeError, "another downloader"):
                    with download.download_lock(root):
                        self.fail("lock should be held")
            finally:
                child.kill()
                child.communicate(timeout=5)
            with download.download_lock(root):
                self.assertTrue((root / ".pull.lock").exists())
