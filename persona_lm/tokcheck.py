"""Read-only check: is the frozen tokenizer still the one everything uses, and has the file changed?

  python -m persona_lm tok-check             # the report
  python -m persona_lm tok-check --reseal    # accept the CURRENT tokenizer as the frozen one (only after a deliberate change)

It looks at:
  * the frozen tokenizer file (work\\tok.json, else tok.json): size, vocabulary, fingerprint, last-modified time
  * a seal file (work\\tok.seal.json) written the first time you run this. Later runs say UNCHANGED or CHANGED against it
  * every token folder it can find (meta.json + train.bin): same tokenizer / DIFFERENT / unknown (made before fingerprints)
  * every checkpoint it can find (ckpt.npz): which tokenizer that model was trained with, and at which step
  * every project.json: the tokenizer fingerprint recorded in the project
It never changes, moves or deletes your tokenizer, data or models. The only file it may write is the seal.
"""
import json
import os
import time

SKIP = {".git", ".venv", "__pycache__", "node_modules", "public_text", "public_clean", "public_mix", "pex_raw", "fineweb2", "tok_sample"}
TOKENIZERS = (os.path.join("work", "tok.json"), "tok.json")


def find_tokenizer(given=None):
    for p in ([given] if given else list(TOKENIZERS)):
        if p and os.path.exists(p):
            return p
    return None


def _walk(root=".", depth=3):
    root = os.path.normpath(root)
    base = root.count(os.sep)
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP and not x.endswith(".part")]
        if d.count(os.sep) - base >= depth:
            dirs[:] = []
        yield d, files


def _short(fp):
    return (fp or "")[:12] + "..." if fp else "-"


def _state(got, frozen):
    if not got:
        return "unknown"
    return "same" if got == frozen else "DIFFERENT"


def read_ckpt_meta(path):
    """Only the small `meta` entry of the .npz is read, not the model weights."""
    import numpy as np
    z = np.load(path)
    return json.loads(bytes(z["meta"].tolist()).decode("utf-8"))


def scan(frozen_fp, root="."):
    rows = []      # (kind, path, state, detail)
    for d, files in _walk(root):
        if "meta.json" in files and "train.bin" in files:
            try:
                meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
                tr = os.path.getsize(os.path.join(d, "train.bin")) // 2
                rows.append(("token folder", d, _state(meta.get("tokenizer_sha256"), frozen_fp), "%s tokens" % format(tr, ",")))
            except Exception as e:
                rows.append(("token folder", d, "unreadable", str(e)[:50]))
        for name in files:
            full = os.path.join(d, name)
            if name == "ckpt.npz":
                try:
                    m = read_ckpt_meta(full)
                    step = m.get("step")
                    rows.append(("checkpoint", d, _state(m.get("tokenizer_sha256"), frozen_fp), "step %s" % (format(step, ",") if isinstance(step, int) else "?")))
                except Exception as e:
                    rows.append(("checkpoint", d, "unreadable", str(e)[:50]))
            elif name == "project.json":
                try:
                    pj = json.load(open(full, encoding="utf-8"))
                    rows.append(("project", d, _state(pj.get("tokenizer_sha256"), frozen_fp), "settings file"))
                except Exception as e:
                    rows.append(("project", d, "unreadable", str(e)[:50]))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def run(reseal=False, tokenizer=None, say=print, root="."):
    from .bpe import Tokenizer
    path = find_tokenizer(tokenizer)
    if not path:
        say("No frozen tokenizer found (looked for %s)." % ", ".join(TOKENIZERS))
        return 2
    tok = Tokenizer.load(path)
    fp = tok.fingerprint()
    st = os.stat(path)
    say("Frozen tokenizer: %s" % path)
    say("  vocabulary: %d tokens   fingerprint: %s   size: %.0f KB   last changed: %s"
        % (tok.vocab_size, fp[:16], st.st_size / 1024, time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))))
    seal_path = os.path.join(os.path.dirname(path) or ".", "tok.seal.json")
    problems = 0
    if os.path.exists(seal_path) and not reseal:
        seal = json.load(open(seal_path, encoding="utf-8"))
        if seal.get("fingerprint") == fp:
            say("  SEAL: UNCHANGED since %s. The tokenizer is still the frozen one." % seal.get("sealed_at", "?"))
        else:
            problems += 1
            say("  SEAL: CHANGED! Sealed %s (%s) but the file now has %s." % (_short(seal.get("fingerprint")), seal.get("sealed_at", "?"), _short(fp)))
            say("        Do not train with it. Put the original tok.json back, or run with --reseal if you changed it on purpose.")
    else:
        with open(seal_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fp, "vocab_size": tok.vocab_size, "sealed_at": time.strftime("%Y-%m-%d %H:%M")}, f)
        say("  SEAL: %s (%s). From now on this check tells you if the file changes."
            % ("re-recorded" if reseal else "first check, current tokenizer recorded", seal_path))
    rows = scan(fp, root)
    say("")
    if not rows:
        say("No token folders, checkpoints or project files found here.")
    else:
        w = max(len(r[1]) for r in rows)
        say("  %-13s %-*s  %-10s %s" % ("what", w, "where", "tokenizer", "detail"))
        for kind, where, state, detail in rows:
            say("  %-13s %-*s  %-10s %s" % (kind, w, where, state, detail))
    n = {s: sum(1 for r in rows if r[2] == s) for s in ("same", "DIFFERENT", "unknown", "unreadable")}
    say("")
    say("Summary: %d use the frozen tokenizer, %d use a DIFFERENT one, %d unknown (made before fingerprints), %d unreadable."
        % (n["same"], n["DIFFERENT"], n["unknown"], n["unreadable"]))
    if n["DIFFERENT"]:
        problems += 1
        say("  DIFFERENT means those files/models were made with another tokenizer. Do not mix them with this one:")
        say("  prepare the text again with the frozen tokenizer, or train that model with its own tokenizer.")
    if n["unknown"]:
        say("  unknown = made before fingerprints existed; it cannot be verified, so re-run `prepare` if in doubt.")
    say("RESULT: %s" % ("all good." if not problems else "PROBLEM found, see above."))
    return 1 if problems else 0


def add_arguments(p):
    p.add_argument("--reseal", action="store_true", help="accept the current tokenizer as the frozen one (after a deliberate change)")
    p.add_argument("--tokenizer", default=None, help="default: work\\tok.json, else tok.json")


def cmd(a):
    raise SystemExit(run(a.reseal, a.tokenizer))
