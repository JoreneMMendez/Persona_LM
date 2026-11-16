"""One-file checkpoints, written atomically (a crash or power cut never leaves a half-written file)."""
import json
import os

import numpy as np

FORMAT_VERSION = 2      # 1 = original files (no version field). 2 adds format_version, tokenizer_sha256, backend info. Loading both works.


def save_ckpt(path, params, opt_arrays, meta, keep_prev=True):
    meta = {"format_version": FORMAT_VERSION, **meta}
    arrays = {"p." + k: v for k, v in params.items()}
    arrays.update(opt_arrays)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    if keep_prev and os.path.exists(path):
        os.replace(path, path.replace(".npz", ".prev.npz"))     # keep the last good one as a fallback
    os.replace(tmp, path)


def _load_one(path):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"].tolist()).decode("utf-8"))
    params = {k[2:]: z[k] for k in z.files if k.startswith("p.")}
    opt = {k: z[k] for k in z.files if k.startswith(("m.", "v.")) or k == "opt_t"}
    return params, opt, meta


def load_ckpt(path):
    """Loads path, or its .prev.npz fallback when the main file is missing or damaged. Returns None if neither works."""
    for p in (path, path.replace(".npz", ".prev.npz")):
        if os.path.exists(p):
            try:
                return _load_one(p)
            except Exception:
                continue
    return None
