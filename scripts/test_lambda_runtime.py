"""Exercise the official Rust Runtime API client with the synthetic checkpoint."""
import http.server
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import unittest


@unittest.skipUnless(os.environ.get("DECIDER_LAMBDA_BIN"), "set DECIDER_LAMBDA_BIN to test the compiled host")
class LambdaRuntimeTests(unittest.TestCase):
    def test_resident_engine_reuses_environment_and_rejects_raw_oversize(self):
        root = Path(__file__).resolve().parents[1]
        incoming, outgoing = queue.Queue(), queue.Queue()
        request = (root / "tests/request.json").read_bytes()
        for payload in [request, b" " * (256 * 1024) + request, request]:
            incoming.put(payload)

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                try:
                    body = incoming.get(timeout=1)
                except queue.Empty:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Lambda-Runtime-Aws-Request-Id", str(time.time_ns()))
                self.send_header("Lambda-Runtime-Deadline-Ms", str(int(time.time() * 1000) + 60000))
                self.send_header("Lambda-Runtime-Invoked-Function-Arn", "arn:aws:lambda:us-east-1:123456789012:function:test")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outgoing.put((self.path, json.loads(body)))
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            shutil.copytree(root / "tests/fixtures/tiny", model)
            (model / "revisions.json").write_text('{"synthetic":true}')
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            env = {**os.environ, "AWS_LAMBDA_RUNTIME_API": f"127.0.0.1:{server.server_port}", "AWS_LAMBDA_FUNCTION_NAME": "test", "AWS_LAMBDA_FUNCTION_MEMORY_SIZE": "10240", "AWS_LAMBDA_FUNCTION_VERSION": "$LATEST", "AWS_LAMBDA_LOG_STREAM_NAME": "local-runtime-test", "DECIDER_MODEL_DIR": str(model), "DECIDER_PRECISION": "int8-compute", "DECIDER_REQUIRE_SDOT": "0", "RAYON_NUM_THREADS": "2"}
            process = subprocess.Popen([str(Path(os.environ["DECIDER_LAMBDA_BIN"]).resolve())], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                rows = [outgoing.get(timeout=60) for _ in range(3)]
                self.assertTrue(rows[0][0].endswith("/response"))
                self.assertTrue(rows[1][0].endswith("/error"))
                self.assertIn("raw request", rows[1][1]["errorMessage"])
                self.assertTrue(rows[2][0].endswith("/response"))
                first, second = rows[0][1], rows[2][1]
                self.assertEqual(first["response"], second["response"])
                self.assertTrue(first["metrics"]["cold"])
                self.assertFalse(second["metrics"]["cold"])
                self.assertEqual(second["metrics"]["sequence"], 1)
                self.assertEqual(first["metrics"]["environment"], second["metrics"]["environment"])
                self.assertEqual(first["metrics"]["load_ms"], second["metrics"]["load_ms"])
            finally:
                process.terminate()
                process.communicate(timeout=10)
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
