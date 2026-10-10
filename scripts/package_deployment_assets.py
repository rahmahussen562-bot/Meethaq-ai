"""Split oversized deployment files into independently uploadable LFS parts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "deployment_assets"
CHUNK_DIR = OUTPUT_DIR / "chunks"
CHUNK_SIZE = 6 * 1024 * 1024
ASSETS = (
    (
        ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/model.onnx",
        "onnx-model",
    ),
    ("chroma_db/chroma.sqlite3", "chroma-sqlite"),
)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if CHUNK_DIR.exists():
        shutil.rmtree(CHUNK_DIR)
    CHUNK_DIR.mkdir(parents=True)
    manifest = {"schema_version": 1, "chunk_size": CHUNK_SIZE, "assets": []}

    for relative, prefix in ASSETS:
        source = PROJECT_DIR / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        parts = []
        with source.open("rb") as handle:
            for index, block in enumerate(iter(lambda: handle.read(CHUNK_SIZE), b"")):
                part = CHUNK_DIR / f"{prefix}-{index:04d}.part"
                part.write_bytes(block)
                parts.append({
                    "path": part.relative_to(PROJECT_DIR).as_posix(),
                    "size": len(block),
                    "sha256": hashlib.sha256(block).hexdigest(),
                })
        manifest["assets"].append({
            "target": relative,
            "size": source.stat().st_size,
            "sha256": digest_file(source),
            "parts": parts,
        })

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Packaged {len(manifest['assets'])} assets into {sum(len(a['parts']) for a in manifest['assets'])} parts")


if __name__ == "__main__":
    main()
