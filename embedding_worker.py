"""Private JSON-lines ONNX worker. Cached models only; never performs network I/O."""
from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Prevent unattended native faults from parking indefinitely in a Windows dialog.
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
faulthandler.enable()

def main():
    model_dir = Path(sys.argv[1])
    required = [model_dir / "model.onnx", model_dir / "tokenizer.json"]
    if any(not file.is_file() for file in required):
        print("[Embedding] Cached ONNX model/tokenizer missing; provision model locally at " + str(model_dir), file=sys.stderr, flush=True)
        return 2
    print("[Embedding] Loading cached CPU ONNX model with one thread...", file=sys.stderr, flush=True)
    from native_runtime import configure_native_runtime
    configure_native_runtime()
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
    print("[Embedding] CPU model ready.", file=sys.stderr, flush=True)
    for line in sys.stdin:
        try:
            texts = json.loads(line)["texts"]
            if not isinstance(texts, list) or len(texts) > 8 or any(not isinstance(text, str) for text in texts):
                raise ValueError("Worker expects up to eight text strings per request.")
            encoded = [tokenizer.encode(text) for text in texts]
            ids = np.array([item.ids for item in encoded], dtype=np.int64)
            masks = np.array([item.attention_mask for item in encoded], dtype=np.int64)
            inputs = {"input_ids": ids, "attention_mask": masks, "token_type_ids": np.zeros_like(ids)}
            outputs = session.run(None, inputs)[0]
            mask = np.broadcast_to(masks[..., None], outputs.shape)
            pooled = np.sum(outputs * mask, axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            vectors = (pooled / np.clip(norm, 1e-12, None)).astype(np.float32)
            print(json.dumps({"embeddings": vectors.tolist()}, allow_nan=False), flush=True)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": "ONNX inference failed; inspect worker diagnostics."}), flush=True)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)

