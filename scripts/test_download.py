import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("download", Path(__file__).with_name("download-model.py"))
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

GOOD = b"pinned weights"
DIGEST = hashlib.sha256(GOOD).hexdigest()


class FakeCurl:
    """Stands in for `curl -C -`: appends the remaining bytes to an existing file."""

    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.calls = 0

    def __call__(self, command, check):
        self.calls += 1
        body = self.bodies.pop(0)
        if isinstance(body, Exception):
            raise body
        target = Path(command[command.index("-o") + 1])
        existing = target.read_bytes() if target.exists() else b""
        target.write_bytes(existing + body[len(existing):])


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.dest = self.root / "lora" / "adapter.safetensors"
        self.part = self.dest.with_name(self.dest.name + ".part")

    def fetch(self, curl):
        with mock.patch.object(download.subprocess, "run", curl):
            download.fetch("repo", "rev", "adapter.safetensors", self.dest, DIGEST)

    def test_a_corrupt_partial_file_is_discarded_and_downloaded_again(self):
        self.part.parent.mkdir(parents=True)
        self.part.write_bytes(b"stale bytes")
        curl = FakeCurl([GOOD, GOOD])
        self.fetch(curl)
        self.assertEqual(self.dest.read_bytes(), GOOD)
        self.assertFalse(self.part.exists())
        self.assertEqual(curl.calls, 2)

    def test_a_failed_resume_starts_over(self):
        self.part.parent.mkdir(parents=True)
        self.part.write_bytes(b"x" * 100)
        curl = FakeCurl([subprocess.CalledProcessError(33, "curl"), GOOD])
        self.fetch(curl)
        self.assertEqual(self.dest.read_bytes(), GOOD)

    def test_a_mismatched_destination_is_replaced(self):
        self.dest.parent.mkdir(parents=True)
        self.dest.write_bytes(b"tampered")
        with mock.patch("sys.stderr"):
            self.fetch(FakeCurl([GOOD]))
        self.assertEqual(self.dest.read_bytes(), GOOD)

    def test_a_persistent_mismatch_fails_without_leaving_a_partial_file(self):
        with self.assertRaisesRegex(RuntimeError, "fresh download"):
            self.fetch(FakeCurl([b"wrong", b"wrong"]))
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.part.exists())

    def test_a_matching_destination_is_not_downloaded(self):
        self.dest.parent.mkdir(parents=True)
        self.dest.write_bytes(GOOD)
        curl = FakeCurl([])
        self.fetch(curl)
        self.assertEqual(curl.calls, 0)


@unittest.skipIf(download.fcntl is None, "flock is POSIX-only")
class LockTests(unittest.TestCase):
    def test_a_leftover_lock_file_does_not_block_a_new_run(self):
        root = Path(tempfile.mkdtemp())
        (root / ".pull.lock").write_text("12345")
        with download.exclusive(root):
            pass

    def test_a_concurrent_run_is_refused(self):
        root = Path(tempfile.mkdtemp())
        with download.exclusive(root):
            with self.assertRaisesRegex(SystemExit, "already running"):
                with download.exclusive(root):
                    pass
        with download.exclusive(root):
            pass


if __name__ == "__main__":
    unittest.main()
