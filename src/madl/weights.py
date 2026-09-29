"""Integrity verification for the separate Hugging Face weight release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weight_manifest(root: str | Path) -> list[str]:
    root = Path(root)
    manifest_path = root / "MANIFEST.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    failures: list[str] = []
    for record in manifest.get("files", []):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe manifest path: {relative}")
            continue
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative.as_posix()}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            failures.append(f"size mismatch: {relative.as_posix()}")
            continue
        if _sha256(path) != str(record["sha256"]).lower():
            failures.append(f"sha256 mismatch: {relative.as_posix()}")
    return failures
