"""Load a provisioned Microsoft C++ runtime before any native vector library."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading

PROJECT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_DIR / ".cache/native-runtime/dll"
RUNTIME_FILES = frozenset({"concrt140.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vcamp140.dll", "vccorlib140.dll",
    "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll"})
_LOCK = threading.Lock()
_HANDLES = []
_SIGNATURE = None


def configure_native_runtime():
    global _SIGNATURE
    with _LOCK:
        if _SIGNATURE is not None:
            return _SIGNATURE
        if os.name != "nt" or not RUNTIME_DIR.exists():
            _SIGNATURE = {"mode": "system", "platform": os.name}
            return _SIGNATURE
        manifest = json.loads((RUNTIME_DIR / "provisioned.json").read_text(encoding="utf-8-sig"))
        hashes = manifest.get("files", {})
        if manifest.get("schema_version") != 1 or set(hashes) != RUNTIME_FILES:
            raise RuntimeError("Microsoft runtime cache is incomplete; rerun provision_native_runtime.ps1.")
        try:
            version = tuple(int(part) for part in manifest["runtime_version"].split("."))
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError("Invalid provisioned Microsoft runtime version.") from exc
        if version < (14, 44, 0, 0):
            raise RuntimeError("Native vector libraries require Microsoft runtime 14.44 or later.")
        for name in sorted(RUNTIME_FILES):
            path = RUNTIME_DIR / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("A provisioned Microsoft runtime DLL is missing or redirected.")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != hashes[name]:
                raise RuntimeError("A provisioned Microsoft runtime DLL changed; rerun runtime setup.")
        import ctypes
        _HANDLES.append(os.add_dll_directory(str(RUNTIME_DIR)))
        # Explicit loading happens before Chroma, HNSW, NumPy or ONNX imports.
        _HANDLES.append(ctypes.WinDLL(str(RUNTIME_DIR / "msvcp140.dll")))
        _SIGNATURE = {"mode": "project", "runtime_version": manifest["runtime_version"],
                      "files": dict(sorted(hashes.items()))}
        return _SIGNATURE
