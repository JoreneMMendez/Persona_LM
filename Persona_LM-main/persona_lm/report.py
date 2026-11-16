"""Read-only reports (Stage 3): `data-report` (what is in a token folder) and `status` (what exists in a work folder, what to do next).

Nothing here changes any file, and nothing here needs PyTorch. The verdicts are RULES OF THUMB, not measurements: they are
there so a beginner gets a plain-words warning before spending hours on a run that cannot work well.
"""
import json
import os

import numpy as np

from .bpe import EOS, Tokenizer
from .model import PRESETS, preset

_CHUNK = 1 << 24
SKIP_DIRS = {".git", "__pycache__", "node_modules", "private"}      # "private" holds real names: never even open it


def _scan(path, dtype, fn):
    """Apply fn to a big file in chunks and add up the results (so a multi-GB token file never has to fit in RAM)."""
    total = 0
    with open(path, "rb") as f:
        while True:
            buf = np.fromfile(f, dtype=dtype, count=_CHUNK)
            if buf.size == 0:
                break
            total += int(fn(buf))
    return total


def data_summary(data_dir):
    """Facts about a folder made by `prepare`. Raises FileNotFoundError with a plain message if it is not one."""
    meta_path = os.path.join(data_dir, "meta.json")
    train_path = os.path.join(data_dir, "train.bin")
    if not (os.path.exists(meta_path) and os.path.exists(train_path)):
        raise FileNotFoundError("%s is not a token folder (no meta.json / train.bin). It is the --out folder of `prepare`." % data_dir)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    s = {"kind": meta.get("kind", "?"), "vocab_size": meta.get("vocab_size"), "tokenizer_sha256": meta.get("tokenizer_sha256")}
    s["train_tokens"] = os.path.getsize(train_path) // 2
    val_path = os.path.join(data_dir, "val.bin")
    s["val_tokens"] = os.path.getsize(val_path) // 2 if os.path.exists(val_path) else 0
    s["documents"] = _scan(train_path, np.uint16, lambda b: (b == EOS).sum())
    mask_path = os.path.join(data_dir, "train_mask.bin")
    s["loss_tokens"] = _scan(mask_path, np.uint8, lambda b: b.sum()) if os.path.exists(mask_path) else None
    return s


def _pretrain_verdict(tokens_per_param):
    if tokens_per_param < 1:
        return "FAR TOO LITTLE text to pretrain this size: the model would mostly memorise it. Add much more text or use a smaller size."
    if tokens_per_param < 10:
        return "Little text for this size. Expect weak language skills; more text helps more than longer training."
    if tokens_per_param < 20:
        return "Below the usual rule of thumb (about 20 tokens per parameter) but usable for a small model."
    return "Reaches the usual rule of thumb (about 20 tokens per parameter)."


def _chat_verdict(loss_tokens):
    if loss_tokens < 50_000:
        return "VERY LITTLE chat text to learn from (under 50k trained tokens): at most a rough hint of your style, and only on top of a pretrained model."
    if loss_tokens < 500_000:
        return "Small, but workable for fine-tuning the STYLE of a model that is already pretrained. Not enough to train from scratch."
    return "A decent amount for fine-tuning a style. Still not enough to train from scratch."


def format_data_report(data_dir, size="s10m", ctx=None, batch=None, accum=None, steps=None, tokenizer_path=None):
    """Lines of plain text describing a token folder and whether it fits the chosen model size."""
    s = data_summary(data_dir)
    from .train import DEFAULT_HP
    batch = batch or DEFAULT_HP["batch"]
    accum = accum or DEFAULT_HP["accum"]
    cfg = preset(size, s["vocab_size"])
    if ctx:
        cfg.ctx = ctx
    lines = ["Token folder %s  (kind: %s)" % (data_dir, s["kind"])]
    lines.append("  training tokens:   %12s   (%d %s)" % ("{:,}".format(s["train_tokens"]), s["documents"], "examples" if s["kind"] == "chat" else "documents"))
    lines.append("  validation tokens: %12s" % "{:,}".format(s["val_tokens"]))
    if s["documents"]:
        lines.append("  average length:    %12.0f tokens per %s" % (s["train_tokens"] / s["documents"], "example" if s["kind"] == "chat" else "document"))
    if s["loss_tokens"] is not None:
        share = 100.0 * s["loss_tokens"] / max(s["train_tokens"], 1)
        lines.append("  tokens that count for learning (loss): %s (%.0f%% of the training tokens; the rest is context the model only reads)" % ("{:,}".format(s["loss_tokens"]), share))
    if s["val_tokens"] <= 2 * cfg.ctx:
        lines.append("  WARNING: validation data is smaller than 2 windows of %d tokens, so validation loss will NOT be computed." % cfg.ctx)
    if tokenizer_path:
        got = Tokenizer.load(tokenizer_path).fingerprint()
        if not s["tokenizer_sha256"]:
            lines.append("  tokenizer check: these files were made before fingerprints existed: cannot verify. Re-run `prepare` to be safe.")
        elif got == s["tokenizer_sha256"]:
            lines.append("  tokenizer check: OK, made with %s" % tokenizer_path)
        else:
            lines.append("  tokenizer check: **DIFFERENT** tokenizer from %s. Do not train with this data and that tokenizer; run `prepare` again." % tokenizer_path)
    params = cfg.n_params()
    lines.append("Model %s: %.1fM parameters, window %d tokens, vocabulary %d" % (size, params / 1e6, cfg.ctx, cfg.vocab_size))
    measure = s["loss_tokens"] if s["loss_tokens"] is not None else s["train_tokens"]
    lines.append("  trained tokens per parameter: %.2f" % (measure / params))
    lines.append("  Verdict (a rule of thumb, not a measurement): " + (_chat_verdict(s["loss_tokens"]) if s["loss_tokens"] is not None else _pretrain_verdict(measure / params)))
    if steps:
        per_step = batch * accum * cfg.ctx
        lines.append("  %d steps of %d x %d sequences of %d tokens = %.1fM tokens read = about %.1f passes over this data (windows are random, so this is approximate)" % (
            steps, batch, accum, cfg.ctx, steps * per_step / 1e6, steps * per_step / max(s["train_tokens"], 1)))
    return lines


# ---------------------------------------------------------------- status of a work folder

def read_ckpt_meta(path):
    """Only the small meta block of a checkpoint (does not load the weights). Falls back to the .prev file like load_ckpt. None if unreadable."""
    for p in (path, path.replace(".npz", ".prev.npz")):
        if os.path.exists(p):
            try:
                with np.load(p) as z:
                    return json.loads(bytes(z["meta"].tolist()).decode("utf-8"))
            except Exception:
                continue
    return None


def validation_trend(metrics_path):
    """(best_val, best_step, last_val, last_step) from a run's metrics.jsonl, or None if there are no validation numbers."""
    best = last = None
    try:
        with open(metrics_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "val_loss" in r:
                    last = (r["val_loss"], r["step"])
                    if best is None or r["val_loss"] < best[0]:
                        best = last
    except OSError:
        return None
    return None if best is None else (best[0], best[1], last[0], last[1])


def _is_tokenizer_file(path):
    try:
        if os.path.getsize(path) > 20_000_000:
            return False
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        return isinstance(j, dict) and "merges" in j and "specials" in j
    except (OSError, ValueError, UnicodeDecodeError):
        return False


def scan_workspace(root, max_depth=2):
    """Finds what exists under root (two folder levels deep): tokenizers, token folders, runs, Messenger imports."""
    found = {"tokenizers": [], "data": [], "runs": [], "messenger": []}
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")) if depth < max_depth else []
        names = set(filenames)
        if "meta.json" in names and "train.bin" in names:
            found["data"].append(dirpath)
        if "ckpt.npz" in names or "ckpt.prev.npz" in names:
            found["runs"].append(dirpath)
        if "stats.json" in names and "train.jsonl" in names:
            found["messenger"].append(dirpath)
        for n in sorted(names):
            if n.endswith(".json") and n not in ("meta.json", "stats.json", "project.json") and _is_tokenizer_file(os.path.join(dirpath, n)):
                found["tokenizers"].append(os.path.join(dirpath, n))
    return found


def format_status(root):
    root = os.path.abspath(root)
    found = scan_workspace(root)
    rel = lambda p: os.path.relpath(p, root) or "."
    lines = ["Work folder: %s" % root]
    lines.append("Messenger imports : " + (", ".join(rel(p) for p in found["messenger"]) or "none yet"))
    lines.append("Tokenizers        : " + (", ".join(rel(p) for p in found["tokenizers"]) or "none yet"))
    for p in found["tokenizers"]:
        lines.append("    %s fingerprint %s... (%d tokens)" % (rel(p), Tokenizer.load(p).fingerprint()[:12], Tokenizer.load(p).vocab_size))
    lines.append("Token folders     : " + (", ".join(rel(p) for p in found["data"]) or "none yet"))
    for p in found["data"]:
        try:
            s = data_summary(p)
            lines.append("    %s: %s, %s train tokens, %s validation tokens" % (rel(p), s["kind"], "{:,}".format(s["train_tokens"]), "{:,}".format(s["val_tokens"])))
        except (OSError, ValueError) as e:
            lines.append("    %s: unreadable (%s)" % (rel(p), e))
    lines.append("Training runs     : " + (", ".join(rel(p) for p in found["runs"]) or "none yet"))
    for p in found["runs"]:
        meta = read_ckpt_meta(os.path.join(p, "ckpt.npz"))
        if meta is None:
            lines.append("    %s: checkpoint unreadable" % rel(p))
            continue
        cfg = meta.get("cfg", {})
        lines.append("    %s: step %s of %s, window %s, vocabulary %s, saved by %s" % (
            rel(p), meta.get("step"), meta.get("hp", {}).get("steps"), cfg.get("ctx"), cfg.get("vocab_size"), (meta.get("backend") or {}).get("name", "?")))
        tr = validation_trend(os.path.join(p, "metrics.jsonl"))
        if tr:
            best, best_step, last, last_step = tr
            lines.append("      validation loss: best %.4f at step %d, latest %.4f at step %d" % (best, best_step, last, last_step))
            if last > best * 1.02 and last_step > best_step:
                lines.append("      WARNING: validation loss has risen since step %d. The model is starting to memorise. Stop training; early stopping is NOT automatic, "
                             "and only the newest checkpoint is kept (plus one previous), so for the next run use fewer --steps." % best_step)
    if os.path.exists(os.path.join(root, "project.json")):
        lines.append("Project file      : project.json found. Next: python -m persona_lm pipeline plan --dir \"%s\"" % root)
    else:
        lines.append("Project file      : none. After you answer the language / window-size questions: python -m persona_lm pipeline init --dir \"%s\" ..." % root)
    return lines


__all__ = ["data_summary", "format_data_report", "format_status", "read_ckpt_meta", "scan_workspace", "validation_trend", "PRESETS"]
