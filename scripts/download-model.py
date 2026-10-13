#!/usr/bin/env python3
"""Fetch the pinned Strands v21 adapter/head and its pinned Qwen3.5 base.

No Python inference dependencies. Downloads are explicit, resumable, and atomic;
the native runtime never downloads anything during inference or model load.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

DECIDER = "StrandsAgents/strands-decider-2B-hobson-v21"
REVISION = "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5"
BASE = "Qwen/Qwen3.5-2B-Base"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def fetch(repo, rev, name, dest, digest=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if digest is None or sha256(dest) == digest:
            return
        raise RuntimeError(f"checksum mismatch: {dest}")
    scratch = dest.with_name(dest.name + ".part")
    subprocess.run(["curl", "-fL", "--retry", "2", "-C", "-", "-o", str(scratch),
                    f"https://huggingface.co/{repo}/resolve/{rev}/{name}"], check=True)
    if digest and sha256(scratch) != digest:
        raise RuntimeError(f"checksum mismatch: {scratch}")
    scratch.replace(dest)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    # Multiple writers must not share a resumable .part file.
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".pull.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
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
    finally:
        os.close(fd)
        lock.unlink()


if __name__ == "__main__":
    main()
