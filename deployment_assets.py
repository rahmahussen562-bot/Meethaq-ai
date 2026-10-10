"""Reassemble deployment assets that are stored as bounded Git LFS parts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "deployment_assets" / "manifest.json"


def _project_path(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"Invalid deployment asset path: {relative!r}")
    resolved = (PROJECT_DIR / candidate).resolve()
    if resolved != PROJECT_DIR and PROJECT_DIR not in resolved.parents:
        raise RuntimeError(f"Deployment asset escapes the project: {relative!r}")
    return resolved


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and _digest(path) == expected_sha256
    )


def ensure_packaged_assets() -> None:
    """Atomically rebuild any missing packaged asset before backend imports."""
    if not MANIFEST_PATH.is_file():
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported deployment asset manifest")

    for asset in manifest.get("assets", []):
        target = _project_path(asset["target"])
        expected_size = int(asset["size"])
        expected_sha256 = asset["sha256"]
        if _matches(target, expected_size, expected_sha256):
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.",
                suffix=".assembling", delete=False,
            ) as output:
                temporary_name = output.name
                combined = hashlib.sha256()
                total = 0
                for part in asset["parts"]:
                    part_path = _project_path(part["path"])
                    if not _matches(part_path, int(part["size"]), part["sha256"]):
                        raise RuntimeError(f"Deployment asset part failed validation: {part['path']}")
                    with part_path.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(block)
                            combined.update(block)
                            total += len(block)
                output.flush()
                os.fsync(output.fileno())
            if total != expected_size or combined.hexdigest() != expected_sha256:
                raise RuntimeError(f"Reassembled deployment asset failed validation: {asset['target']}")
            os.replace(temporary_name, target)
            temporary_name = None
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
