"""Explicit, checksum-verified setup for the locally cached Chroma ONNX model.

No network operation occurs unless the operator supplies --download. For an
air-gapped installation, supply the official archive with --archive instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
from urllib.request import urlopen

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DOWNLOAD_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
MODEL_FILES = frozenset({"config.json", "model.onnx", "special_tokens_map.json",
                         "tokenizer_config.json", "tokenizer.json", "vocab.txt"})
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_EXTRACTED_BYTES = 128 * 1024 * 1024
DISK_RESERVE_BYTES = 64 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_DEADLINE_SECONDS = 600
CHUNK_BYTES = 1024 * 1024


def cache_root() -> Path:
    project = PROJECT_DIR.resolve()
    root = (project / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2").resolve()
    if not root.is_relative_to(project):
        raise ValueError("The model cache must remain inside the project directory.")
    return root


def require_space(path: Path, required: int) -> None:
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < required + DISK_RESERVE_BYTES:
        raise OSError("Insufficient disk space for model setup and the 64 MiB reserve.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(archive: Path) -> None:
    if not archive.is_file() or not 0 < archive.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("The ONNX archive is missing, empty, or exceeds the size limit.")
    if file_sha256(archive) != MODEL_SHA256:
        raise ValueError("The ONNX archive SHA-256 does not match the official Chroma model.")


def download_archive(root: Path) -> Path:
    require_space(root, 0)
    root.mkdir(parents=True, exist_ok=True)
    partial = None
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
    try:
        with urlopen(MODEL_DOWNLOAD_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            if response.geturl() != MODEL_DOWNLOAD_URL:
                raise ValueError("The model download redirected away from the official URL.")
            advertised = response.headers.get("Content-Length")
            length = int(advertised) if advertised is not None else None
            if length is not None and not 0 < length <= MAX_ARCHIVE_BYTES:
                raise ValueError("The model download exceeds the archive size limit.")
            budget = length if length is not None else MAX_ARCHIVE_BYTES
            # Extraction has its own exact-size preflight after the archive is verified.
            require_space(root, budget)
            descriptor, name = tempfile.mkstemp(prefix="download-", suffix=".tar.gz.part", dir=root)
            partial = Path(name)
            with os.fdopen(descriptor, "wb") as target:
                received = 0
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("The model download exceeded its ten-minute deadline.")
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    if received + len(block) > budget:
                        raise ValueError("The model download exceeds the archive size limit.")
                    require_space(root, budget - received)
                    target.write(block)
                    received += len(block)
                if length is not None and received != length:
                    raise ValueError("The model download ended before the advertised size.")
                target.flush()
                os.fsync(target.fileno())
        verify_archive(partial)
        return partial
    except BaseException:
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise


def archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    files: dict[str, tarfile.TarInfo] = {}
    total_size = 0
    count = 0
    for member in archive:
        count += 1
        if count > 32:
            raise ValueError("The model archive contains too many entries.")
        name = member.name
        while name.startswith("./"):
            name = name[2:]
        if "\\" in name or name.startswith("/") or ".." in name.split("/"):
            raise ValueError("Unsafe path in the model archive.")
        if member.isdir() and name.rstrip("/") in {"", ".", "onnx"}:
            continue
        parts = name.split("/")
        if (not member.isfile() or len(parts) != 2 or parts[0] != "onnx"
                or parts[1] not in MODEL_FILES):
            raise ValueError("The model archive contains an unexpected file, link, or directory.")
        filename = parts[1]
        if filename in files or member.size <= 0:
            raise ValueError("The model archive contains duplicate or empty files.")
        total_size += member.size
        if total_size > MAX_EXTRACTED_BYTES:
            raise ValueError("The model archive exceeds the extracted size limit.")
        files[filename] = member
    if set(files) != MODEL_FILES:
        raise ValueError("The model archive does not contain every required ONNX file.")
    return files


def cached_model(root: Path) -> Path | None:
    model_dir = root / "onnx"
    if model_dir.is_symlink() or not model_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("The model cache directory must not redirect outside the project cache.")
    if not model_dir.exists():
        return None
    manifest = model_dir / "provisioned.json"
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if document.get("archive_sha256") != MODEL_SHA256:
            raise ValueError("The existing model cache has an unverified archive identity.")
        recorded = document["files"]
        if set(recorded) != MODEL_FILES:
            raise ValueError("The existing model cache manifest is incomplete.")
        for filename in MODEL_FILES:
            path = model_dir / filename
            if not path.is_file() or path.is_symlink() or file_sha256(path) != recorded[filename]:
                raise ValueError("The existing model cache has missing or changed files.")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("The existing model cache is incomplete or unverified.") from error
    return model_dir.resolve()


def install_archive(archive_path: Path, root: Path) -> Path:
    # Authenticate all archive bytes before opening it or creating extraction files.
    verify_archive(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        files = archive_members(archive)
        existing = cached_model(root)
        if existing is not None:
            return existing
        remaining_bytes = sum(member.size for member in files.values())
        require_space(root, remaining_bytes)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="extract-", dir=root) as stage_name:
            stage = Path(stage_name)
            model_dir = stage / "onnx"
            model_dir.mkdir()
            hashes = {}
            for filename, member in files.items():
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("The model archive contains an unreadable file.")
                destination = model_dir / filename
                with source, destination.open("xb") as target:
                    for block in iter(lambda: source.read(CHUNK_BYTES), b""):
                        require_space(root, remaining_bytes)
                        target.write(block)
                        remaining_bytes -= len(block)
                    target.flush()
                    os.fsync(target.fileno())
                if destination.stat().st_size != member.size:
                    raise ValueError("The model archive contains a truncated file.")
                hashes[filename] = file_sha256(destination)
            require_space(root, 4096)
            (model_dir / "provisioned.json").write_text(json.dumps({
                "archive_sha256": MODEL_SHA256, "files": hashes,
            }, sort_keys=True), encoding="utf-8")
            # Publish the complete directory once; a failed extraction never becomes active.
            model_dir.rename(root / "onnx")
    return (root / "onnx").resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Download the official model explicitly.")
    mode.add_argument("--archive", type=Path, help="Import the official onnx.tar.gz without network access.")
    arguments = parser.parse_args(argv)
    downloaded = None
    try:
        root = cache_root()
        existing = cached_model(root)
        if existing is not None:
            print(existing, flush=True)
            return 0
        if not arguments.download and arguments.archive is None:
            raise ValueError("No cached model exists. Run --download or --archive <onnx.tar.gz> during setup.")
        archive = arguments.archive
        if arguments.download:
            downloaded = download_archive(root)
            archive = downloaded
        model_dir = install_archive(archive.resolve(), root)
        print(model_dir, flush=True)
        return 0
    except (OSError, ValueError, tarfile.TarError) as error:
        print("[Model setup] " + str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        if downloaded is not None:
            downloaded.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
