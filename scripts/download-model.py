#!/usr/bin/env python3
"""Fetch the pinned Strands v21 adapter/head and its pinned Qwen3.5 base.

No Python inference dependencies. Downloads are explicit, resumable, and atomic;
the native runtime never downloads anything during inference or model load.
"""
import argparse
import fcntl
from contextlib import contextmanager
import uuid
import hashlib
import json
import os
from pathlib import Path
import subprocess

DECIDER = "StrandsAgents/strands-decider-2B-hobson-v21"
REVISION = "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5"
BASE = "Qwen/Qwen3.5-2B-Base"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
# Raw metadata at the immutable revisions above. Bootstrap files must be checked
# before parsing them; otherwise a corrupt manifest/provenance would block repair.
METADATA_SHA256 = {
    "strands_decider_config.json": "9b660cabb91ab9683dd62dee83e00a5ca30dff8657489e6ce473136f8d9eb529",
    "provenance.json": "2fecee3c08fe0d4682f5d4563c55bc89f7104f97e2780d85e9fc9ae8881e1c28",
    "MANIFEST.sha256": "097556a1111eb686e43ba4c4a756f57a4ab053328570f5ec28937e9403111992",
    "base/config.json": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    "base/model.safetensors.index.json": "74d2ddfe79f10f35b27b498632f02b97b60dd9ec39b35d7c5a890c399284e319",
    "base/LICENSE": "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
}



def fetch(repo, rev, name, dest, digest=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if digest is None or sha256(dest) == digest:
            return
        # Preserve corrupt bytes for diagnosis without blocking the normal command.
        dest.replace(dest.with_name(dest.name + ".corrupt-" + uuid.uuid4().hex))
    scratch = dest.with_name(dest.name + ".part")
    for attempt in range(2):
        try:
            subprocess.run(["curl", "-fL", "--retry", "2", "-C", "-", "-o", str(scratch),
                            f"https://huggingface.co/{repo}/resolve/{rev}/{name}"], check=True)
        except subprocess.CalledProcessError as error:
            # A completed/bad prefix can cause HTTP 416 or a rejected range before
            # curl reaches the digest check. Retry such resumes once without it.
            if attempt == 0 and scratch.exists() and error.returncode in (22, 33, 36):
                scratch.unlink()
                continue
            raise
        if digest is None or sha256(scratch) == digest:
            scratch.replace(dest)
            return
        # A resumed prefix may be corrupt. Never resume it again: retry once from zero.
        scratch.unlink()
    raise RuntimeError(f"checksum mismatch after a clean retry: {dest}; partial bytes removed, rerun the same command")


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@contextmanager
def download_lock(root):
    # Multiple writers must not share a resumable .part file.
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".pull.lock"
    fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another downloader is using {root}; wait for it to finish") from error
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        # Keep the inode: unlinking permits a second writer to lock a new inode.
        # The kernel releases flock even after SIGKILL; stale PID text is harmless.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    with download_lock(root):
        files = ["strands_decider_config.json", "provenance.json", "MANIFEST.sha256",
                 "LICENSE.md", "tokenizer.json", "tokenizer_config.json",
                 "head.safetensors", "lora/adapter_config.json", "lora/adapter_model.safetensors"]
        for name in files[:3]:
            fetch(DECIDER, REVISION, name, root / name, METADATA_SHA256[name])
        provenance = json.loads((root / "provenance.json").read_text())
        if provenance["base_model"] != BASE or provenance["base_model_revision"] != BASE_REVISION:
            raise RuntimeError("unexpected base model provenance")
        digests = {}
        for line in (root / "MANIFEST.sha256").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            digests[name.lstrip("*")] = digest
        for name in files:
            fetch(DECIDER, REVISION, name, root / name, digests.get(name) or METADATA_SHA256[name])
        fetch(BASE, BASE_REVISION, "LICENSE", root / "base/LICENSE", METADATA_SHA256["base/LICENSE"])
        fetch(BASE, BASE_REVISION, "config.json", root / "base/config.json", METADATA_SHA256["base/config.json"])
        fetch(BASE, BASE_REVISION, "model.safetensors.index.json", root / "base/model.safetensors.index.json", METADATA_SHA256["base/model.safetensors.index.json"])
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
