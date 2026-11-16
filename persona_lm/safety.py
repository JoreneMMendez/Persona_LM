"""Stage 5 safety tools shared by both training backends (NumPy and PyTorch).

  * backup_before_training(): copies ckpt.npz (and the tokenizer, if given) into <run>/backups/<time>/ BEFORE a run starts or resumes.
    Nothing is ever deleted or overwritten; an identical backup is not made twice; a failed backup only prints a warning.
  * best checkpoint: whenever the validation loss reaches a new low, the weights are saved to <run>/ckpt.best.npz (weights only,
    no optimizer, so it is a third of the size). It loads for chat/export like any checkpoint. The normal ckpt.npz and its
    ckpt.prev.npz are unchanged, so resuming works exactly as before.
  * early stop: `early_stop` = N stops training after N validation checks in a row without a new best. 0 = off (the default).
All of it is derived from the run's history list, so it survives Ctrl+C and resuming.
"""
import hashlib
import json
import os
import shutil
import time

from .ckpt import save_ckpt

BEST_NAME = "ckpt.best.npz"
BACKUP_DIR = "backups"


# ------------------------------------------------------------------ validation history
def val_records(history):
    return [(r["step"], r["val_loss"]) for r in history if "val_loss" in r]


def best_so_far(history):
    """(step, val_loss) of the lowest validation loss so far, or None. The earliest step wins a tie."""
    best = None
    for step, v in val_records(history):
        if best is None or v < best[1]:
            best = (step, v)
    return best


def is_new_best(history):
    """True when the LAST history record carries a validation loss that is strictly lower than every earlier one."""
    if not history or "val_loss" not in history[-1]:
        return False
    prev = best_so_far(history[:-1])
    return prev is None or history[-1]["val_loss"] < prev[1]


def evals_without_improvement(history):
    """How many validation checks in a row, counted back from the latest, did not beat the best before them."""
    vals = [v for _, v in val_records(history)]
    best, since = None, 0
    for v in vals:
        if best is None or v < best:
            best, since = v, 0
        else:
            since += 1
    return since


def should_early_stop(history, patience):
    return bool(patience) and patience > 0 and evals_without_improvement(history) >= patience


# ------------------------------------------------------------------ best checkpoint
def save_best(run_dir, params, meta):
    """Weights only, written atomically. `params` are NumPy arrays (the torch backend converts first)."""
    path = os.path.join(run_dir, BEST_NAME)
    save_ckpt(path, params, {}, meta, keep_prev=False)
    return path


# ------------------------------------------------------------------ backup before training
def _sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _latest_backup(bdir):
    if not os.path.isdir(bdir):
        return None
    names = sorted(n for n in os.listdir(bdir) if os.path.isfile(os.path.join(bdir, n, "backup.json")))
    if not names:
        return None
    try:
        with open(os.path.join(bdir, names[-1], "backup.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def backup_before_training(run_dir, tokenizer=None, log=print, now=None):
    """Returns the backup folder, or None when nothing needed backing up (new run, identical backup already there, or a problem)."""
    try:
        ck = os.path.join(run_dir, "ckpt.npz")
        have_ck = os.path.isfile(ck)
        have_tok = bool(tokenizer) and os.path.isfile(tokenizer)
        if not have_ck and not have_tok:
            return None
        hashes = {}
        if have_ck:
            hashes["ckpt_sha256"] = _sha(ck)
        if have_tok:
            hashes["tokenizer_sha256"] = _sha(tokenizer)
        bdir = os.path.join(run_dir, BACKUP_DIR)
        last = _latest_backup(bdir)
        if last and all(last.get(k) == v for k, v in hashes.items()):
            return None
        need = sum(os.path.getsize(p) for p, ok in ((ck, have_ck), (tokenizer, have_tok)) if ok)
        os.makedirs(run_dir, exist_ok=True)
        if shutil.disk_usage(run_dir).free < 2 * need:
            log("warning: not enough free disk space for a backup of %s, training goes on without one" % run_dir)
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        out = os.path.join(bdir, stamp)
        k = 1
        while os.path.exists(out):                      # never reuse (and so never overwrite) a backup folder
            k += 1
            out = os.path.join(bdir, "%s_%d" % (stamp, k))
        os.makedirs(out)
        if have_ck:
            shutil.copy2(ck, os.path.join(out, "ckpt.npz"))
        if have_tok:
            shutil.copy2(tokenizer, os.path.join(out, os.path.basename(tokenizer)))
        with open(os.path.join(out, "backup.json"), "w", encoding="utf-8") as f:
            json.dump({"made": stamp, "files": sorted(os.listdir(out)), **hashes}, f)
        log("backup: %s saved to %s" % (" and ".join(n for n, ok in (("checkpoint", have_ck), ("tokenizer", have_tok)) if ok), out))
        return out
    except Exception as e:                              # a backup problem must never stop the training
        log("warning: backup failed (%s: %s), training goes on without one" % (type(e).__name__, e))
        return None
