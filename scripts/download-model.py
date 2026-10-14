#!/usr/bin/env python3
"""Fetch the pinned Strands v21 adapter/head and its pinned Qwen3.5 base.

No Python inference dependencies. Downloads are explicit, resumable, and atomic;
the native runtime never downloads anything during inference or model load.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    import fcntl
except ImportError:  # Windows: fall back to an exclusive-create lock file.
    fcntl = None

DECIDER = "StrandsAgents/strands-decider-2B-hobson-v21"
REVISION = "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5"
BASE = "Qwen/Qwen3.5-2B-Base"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def fetch(repo, rev, name, dest, digest=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if digest is None or sha256(dest) == digest:
            return
        print(f"{dest}: checksum mismatch; downloading it again", file=sys.stderr)
        dest.unlink()
    scratch = dest.with_name(dest.name + ".part")
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
    for attempt in range(2):
        # The first attempt resumes an existing .part; the second always starts over.
        try:
            subprocess.run(["curl", "-fL", "--retry", "2", "-C", "-", "-o", str(scratch), url],
                           check=True)
        except subprocess.CalledProcessError:
            if attempt or not scratch.exists():
                raise
        else:
            if digest is None or sha256(scratch) == digest:
                scratch.replace(dest)
                return
        # A partial file can hold bytes from an interrupted or superseded download.
        scratch.unlink(missing_ok=True)
    raise RuntimeError(f"checksum mismatch after a fresh download of {name} from {repo}@{rev}")


@contextmanager
def exclusive(root):
    """Hold the download lock; the operating system releases it if the process dies."""
    lock = root / ".pull.lock"
    if fcntl is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise SystemExit(f"{lock} exists: another download may be running; "
                             "delete it if not") from None
        try:
            yield
        finally:
            os.close(fd)
            lock.unlink()
        return
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another download is already running in {root}") from None
        yield


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    root.mkdir(parents=True, exist_ok=True)
    # Multiple writers must not share a resumable .part file.
    with exclusive(root):
        files = ["strands_decider_config.json", "provenance.json", "MANIFEST.sha256",
                 "LICENSE.md", "tokenizer.json", "tokenizer_config.json",
                 "head.safetensors", "lora/adapter_config.json", "lora/adapter_model.safetensors"]
        for name in files[:3]:
            fetch(DECIDER, REVISION, name, root / name)
        provenance = json.loads((root / "provenance.json").read_text())
        if provenance["base_model"] != BASE or provenance["base_model_revision"] != BASE_REVISION:
            raise RuntimeError("unexpected base model provenance")
        digests = {}
        for line in (root / "MANIFEST.sha256").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            digests[name.lstrip("*")] = digest
        for name in files:
            fetch(DECIDER, REVISION, name, root / name, digests.get(name))
        fetch(BASE, BASE_REVISION, "LICENSE", root / "base/LICENSE")
        fetch(BASE, BASE_REVISION, "config.json", root / "base/config.json")
        fetch(BASE, BASE_REVISION, "model.safetensors.index.json", root / "base/model.safetensors.index.json")
        index = json.loads((root / "base/model.safetensors.index.json").read_text())
        metadata = json.loads(subprocess.check_output([
            "curl", "-fsSL", "--retry", "2",
            f"https://huggingface.co/api/models/{BASE}/revision/{BASE_REVISION}?blobs=true"
        ]))
        base_hashes = {f["rfilename"]: f.get("lfs", {}).get("sha256") for f in metadata["siblings"]}
        for name in sorted(set(index["weight_map"].values())):
            if Path(name).name != name or name.startswith(".") or "\\" in name:
                raise RuntimeError(f"invalid shard path: {name}")
            digest = base_hashes.get(name)
            if digest is None:
                raise RuntimeError(f"no pinned SHA256 for base shard: {name}")
            fetch(BASE, BASE_REVISION, name, root / "base" / name, digest)
        (root / "revisions.json").write_text(json.dumps({"decider": REVISION, "base": BASE_REVISION}, indent=2) + "\n")
        print(root)


if __name__ == "__main__":
    main()
