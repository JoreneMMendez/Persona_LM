"""`python -m persona_lm doctor`: read-only health check that says in plain words what is wrong and what to do.

  python -m persona_lm doctor                    # look in the current project folder (tokenizer work\\tok.json)
  python -m persona_lm doctor --root C:\\path     # another project folder
  python -m persona_lm doctor --deep             # also read every weight and look for NaN/inf (slower)
  python -m persona_lm doctor --windows          # self-check of this computer: tiny training runs, console, panel, menu, hybrid tests

It checks that the tokenizer loads, that every token folder and every checkpoint was made with THAT tokenizer and the same
vocabulary size, that the files are complete, and how each run's validation loss went (overfitting).
It never changes, moves or deletes anything and never prints chat text (it only reads meta.json files and checkpoint metadata).
"""
import json
import os

from .tokcheck import _walk, find_tokenizer, read_ckpt_meta

OK, WARN, PROBLEM = "ok", "warning", "PROBLEM"


class Report:
    def __init__(self, say=print):
        self.say, self.items = say, []

    def add(self, level, where, msg, fix=None):
        self.items.append((level, where, msg, fix))
        self.say("  [%-7s] %s: %s" % (level, where, msg))
        if fix:
            self.say("            -> %s" % fix)

    def count(self, level):
        return sum(1 for i in self.items if i[0] == level)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_tokenizer(rep, path):
    from .bpe import Tokenizer
    if not path or not os.path.isfile(path):
        rep.add(PROBLEM, "tokenizer", "no tokenizer file found (looked for work\\tok.json and tok.json)",
                "give its path with --tokenizer. Never create a new one: every model depends on the old one")
        return None
    try:
        tok = Tokenizer.load(path)
        fp, vocab = tok.fingerprint(), tok.vocab_size
    except (Exception, SystemExit) as e:           # Tokenizer.load exits with a friendly message for a wrong file
        rep.add(PROBLEM, path, "is not a readable tokenizer (%s)" % type(e).__name__,
                "use the real tokenizer (the file with 'merges' and 'specials'), normally work\\tok.json")
        return None
    rep.add(OK, path, "tokenizer loads: %d tokens, fingerprint %s..." % (vocab, fp[:12]))
    return fp, vocab


def check_data_folder(rep, d, tok):
    fp, vocab = tok if tok else (None, None)
    try:
        meta = _read_json(os.path.join(d, "meta.json"))
    except Exception:
        rep.add(PROBLEM, d, "meta.json cannot be read", "make the token files again with `prepare` (same tokenizer)")
        return
    problems = 0
    got_fp = meta.get("tokenizer_sha256")
    if fp:
        if got_fp and got_fp != fp:
            problems += 1
            rep.add(PROBLEM, d, "these token files were made with a DIFFERENT tokenizer", "do not train on them. Run `prepare` again with the frozen tokenizer")
        elif not got_fp:
            problems += 1
            rep.add(WARN, d, "made before tokenizer fingerprints existed, so it cannot be checked", "if unsure, run `prepare` again with the frozen tokenizer")
        if meta.get("vocab_size") not in (None, vocab):
            problems += 1
            rep.add(PROBLEM, d, "vocabulary size %s does not match the tokenizer (%d)" % (meta.get("vocab_size"), vocab), "run `prepare` again")
    for name, key in (("train.bin", "train_tokens"), ("val.bin", "val_tokens")):
        path = os.path.join(d, name)
        if not os.path.isfile(path):
            if name == "train.bin":
                problems += 1
                rep.add(PROBLEM, d, "train.bin is missing", "run `prepare` again")
            continue
        n = os.path.getsize(path) // 2
        if meta.get(key) is not None and meta[key] != n:
            problems += 1
            rep.add(PROBLEM, d, "%s holds %d tokens but meta.json says %d (file cut short or replaced?)" % (name, n, meta[key]), "run `prepare` again")
    if meta.get("kind") == "chat":
        for name in ("train_mask.bin",):
            if not os.path.isfile(os.path.join(d, name)):
                problems += 1
                rep.add(PROBLEM, d, "chat data without %s (the model would learn from every token, not only your replies)" % name, "run `prepare --kind chat` again")
    if not problems:
        rep.add(OK, d, "token folder is complete%s" % (" and matches the tokenizer" if got_fp and fp else ""))


def _best_and_last(history):
    vals = [(r["step"], r["val_loss"]) for r in history if "val_loss" in r]
    if not vals:
        return None, None
    return min(vals, key=lambda t: t[1]), vals[-1]


def check_run(rep, d, tok, deep=False):
    import numpy as np
    fp, vocab = tok if tok else (None, None)
    main, prev = os.path.join(d, "ckpt.npz"), os.path.join(d, "ckpt.prev.npz")
    meta, used = None, None
    for p in (main, prev):
        if os.path.isfile(p):
            try:
                meta, used = read_ckpt_meta(p), p
                break
            except Exception:
                rep.add(PROBLEM if p == main else WARN, p, "cannot be read (damaged or not a Persona_LM checkpoint)",
                        "training and chat fall back to ckpt.prev.npz when there is one" if p == main else None)
    if meta is None:
        rep.add(PROBLEM, d, "no readable checkpoint", "restore one from %s or from a backups\\ folder" % "ckpt.best.npz" if os.path.isfile(os.path.join(d, "ckpt.best.npz")) else "restore one from a backups\\ folder")
        return
    if used == prev:
        rep.add(WARN, d, "ckpt.npz is missing or damaged, so only the older ckpt.prev.npz can be used", "copy ckpt.prev.npz to ckpt.npz if you want to continue from it")
    cfg = meta.get("cfg", {})
    bad = 0
    if fp:
        if meta.get("tokenizer_sha256") and meta["tokenizer_sha256"] != fp:
            bad += 1
            rep.add(PROBLEM, d, "this model was trained with a DIFFERENT tokenizer than the one checked", "chat with it only using the tokenizer it was trained with")
        elif not meta.get("tokenizer_sha256"):
            rep.add(WARN, d, "old checkpoint without a tokenizer fingerprint, cannot be checked")
        if cfg.get("vocab_size") not in (None, vocab):
            bad += 1
            rep.add(PROBLEM, d, "the model has vocabulary size %s but the tokenizer has %d" % (cfg.get("vocab_size"), vocab), "use the tokenizer this model was trained with")
    best, last = _best_and_last(meta.get("history", []))
    step = meta.get("step", "?")
    steps = (meta.get("hp") or {}).get("steps")
    msg = "step %s%s" % (step, (" of %s" % steps) if steps else "")
    if best and last:
        msg += ", validation loss: best %.3f at step %d, latest %.3f at step %d" % (best[1], best[0], last[1], last[0])
    if best and last and last[1] > best[1] * 1.05 and last[0] > best[0]:
        bad += 1
        has_best = os.path.isfile(os.path.join(d, "ckpt.best.npz"))
        rep.add(WARN, d, msg + ". Validation loss has risen since the best point: the model is overfitting (memorizing your chats)",
                "use ckpt.best.npz in this folder for chat" if has_best else
                "stop earlier next time: train with --early-stop 3 (saves ckpt.best.npz). This run's best step was %d" % best[0])
    elif not bad:
        rep.add(OK, d, msg)
    if deep:
        try:
            z = np.load(used)
            n_bad = [k[2:] for k in z.files if k.startswith("p.") and not np.isfinite(z[k]).all()]
        except Exception as e:
            n_bad = ["(unreadable: %s)" % type(e).__name__]
        if n_bad:
            rep.add(PROBLEM, d, "weights contain NaN/inf in %d array(s), e.g. %s" % (len(n_bad), n_bad[0]), "this checkpoint is broken; go back to ckpt.prev.npz, ckpt.best.npz or a backup")
        else:
            rep.add(OK, d, "all weights are finite numbers")


def scan_folders(root="."):
    data, runs = [], []
    for d, files in _walk(root):
        base = os.path.basename(d)
        if base in ("backups", "_to_delete"):
            continue
        if "meta.json" in files and "train.bin" in files:
            data.append(d)
        if "ckpt.npz" in files or "ckpt.prev.npz" in files:
            runs.append(d)
    return sorted(data), sorted(runs)


def run(root=".", tokenizer=None, deep=False, say=print):
    rep = Report(say)
    path = find_tokenizer(tokenizer) if tokenizer else find_tokenizer(None) or find_tokenizer(os.path.join(root, "work", "tok.json"))
    say("Persona_LM doctor (read-only): %s" % os.path.abspath(root))
    tok = check_tokenizer(rep, path)
    data, runs = scan_folders(root)
    if not data and not runs:
        say("  no token folders or runs found under %s" % root)
    for d in data:
        check_data_folder(rep, d, tok)
    for d in runs:
        check_run(rep, d, tok, deep)
    p, w = rep.count(PROBLEM), rep.count(WARN)
    say("")
    say("Result: %d problem(s), %d warning(s). %s" % (p, w, "Everything checked looks fine." if not p and not w else
                                                        "Fix the PROBLEM lines first; nothing was changed by this check."))
    return 1 if p else 0


def add_arguments(s):
    s.add_argument("--root", default=".", help="the project folder to look in (default: here)")
    s.add_argument("--tokenizer", default=None, help="the frozen tokenizer (default: work\\tok.json, else tok.json)")
    s.add_argument("--deep", action="store_true", help="also read all weights and look for NaN/inf (slower)")
    s.add_argument("--windows", action="store_true", help="instead of the file checks: try tiny training runs, the training panel, a menu form and the hybrid tests on THIS computer (persona_lm/selfcheck.py)")
    s.add_argument("--no-tests", action="store_true", help="(with --windows) do not run tests\\test_hybrid.py")


def cmd(a):
    if getattr(a, "windows", False):
        from . import selfcheck
        code = selfcheck.run(a.root, run_tests=not getattr(a, "no_tests", False))
        if code:
            raise SystemExit(code)
        return
    code = run(a.root, a.tokenizer, a.deep)
    if code:
        raise SystemExit(code)
