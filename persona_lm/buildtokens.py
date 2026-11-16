"""One-click: turn the cleaned language data into token files, ready for training.

  python -m persona_lm build-tokens                 # all languages that are cleaned; tokenizer work\\tok.json; NEW folder data_pre_new
  python -m persona_lm build-tokens --lang en,taglish --out data_pre_mine

What it does, in order (it never deletes or overwrites anything of yours):
  1. finds your tokenizer (work\\tok.json, else tok.json). It is only READ.
  2. for every language whose cleaned file exists (public_clean\\<lang>\\<lang>.jsonl), builds the balanced mix public_mix\\<lang>.jsonl
     if it is missing or older than the cleaned file.
  3. turns those mix files into token files (train.bin / val.bin) in a NEW folder. If the folder name is taken, it uses _2, _3, ...
     The folder appears only when it is finished; a stopped run leaves nothing half-written.
Needs no internet. The menu item "Language data -> token files" runs exactly this.
"""
import os
import shutil
import time

TOKENIZERS = (os.path.join("work", "tok.json"), "tok.json")


class BuildError(Exception):
    pass


def add_arguments(p):
    p.add_argument("--lang", default=None, help="comma list of: en, tl, ceb, taglish (default: every language that is already cleaned)")
    p.add_argument("--tokenizer", default=None, help="default: work\\tok.json, else tok.json")
    p.add_argument("--out", default=None, help="default: a NEW folder data_pre_new (or data_pre_new_2, ...). An existing non-empty folder is never used.")
    p.add_argument("--val-every", type=int, default=50, help="every Nth document goes to validation (default 50)")


def new_folder(want):
    """A folder name that does not exist yet or is empty: want, want_2, want_3, ..."""
    def free(p):
        return not os.path.exists(p) or (os.path.isdir(p) and not os.listdir(p))
    if free(want):
        return want
    n = 2
    while not free("%s_%d" % (want, n)):
        n += 1
    return "%s_%d" % (want, n)


def find_tokenizer(given):
    if given:
        if not os.path.exists(given):
            raise BuildError("tokenizer not found: %s" % given)
        return given
    for p in TOKENIZERS:
        if os.path.exists(p):
            return p
    raise BuildError("no tokenizer found (looked for %s). Train one first, or pass --tokenizer PATH." % ", ".join(TOKENIZERS))


def run(langs=None, tokenizer=None, out=None, val_every=50, say=print):
    from . import corpus as C
    from . import data as D
    from .bpe import Tokenizer
    wanted = C.check_langs(langs) if langs else list(C.LANG_ORDER)
    have = [l for l in wanted if os.path.exists(C.LANGS[l]["clean"]) and os.path.getsize(C.LANGS[l]["clean"]) > 0]
    skipped = [l for l in wanted if l not in have]
    if not have:
        raise BuildError("no cleaned language data found. Run the menu item 'Language data (download)' first.")
    tok_path = find_tokenizer(tokenizer)
    say("Tokenizer: %s (read only)" % tok_path)
    say("Languages: %s" % ", ".join(C.LANGS[l]["label"] for l in have))
    if skipped:
        say("Not included (no cleaned file yet): %s" % ", ".join(C.LANGS[l]["label"] for l in skipped))
    need = []
    for l in have:
        mp = os.path.join("public_mix", l + ".jsonl")
        if not os.path.exists(mp) or os.path.getmtime(mp) < os.path.getmtime(C.LANGS[l]["clean"]):
            need.append(l)
    if need:
        say("Building the mix for: %s" % ", ".join(C.LANGS[l]["label"] for l in need))
        C._mix(need, C._Stop(None))
    files = [os.path.join("public_mix", l + ".jsonl") for l in have]
    for f in files:
        say("  mix  %-28s %7.1f MB" % (f, os.path.getsize(f) / 1e6))
    target = new_folder(out or "data_pre_new")
    if out and os.path.exists(out) and os.listdir(out):
        raise BuildError("%s already has files in it. Pick another --out, or leave it out and a new folder is chosen for you." % out)
    part = target + ".part"
    shutil.rmtree(part, ignore_errors=True)
    say("Turning text into tokens into %s (this takes a while; Ctrl+C is safe)..." % target)
    t0 = time.time()
    try:
        tok = Tokenizer.load(tok_path)
        D.prepare(tok, D.pretrain_docs(files, tok), part, with_mask=False, val_every=val_every, log=lambda m: say("  " + m))
        if not os.path.exists(os.path.join(part, "train.bin")) or os.path.getsize(os.path.join(part, "train.bin")) == 0:
            raise BuildError("no tokens were produced (empty train.bin).")
        if os.path.isdir(target):
            os.rmdir(target)                     # an empty folder only (new_folder guarantees it)
        os.replace(part, target)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    n_train = os.path.getsize(os.path.join(target, "train.bin")) // 2
    n_val = os.path.getsize(os.path.join(target, "val.bin")) // 2
    say("DONE in %d s: %s  (train %d tokens, validation %d tokens)" % (time.time() - t0, target, n_train, n_val))
    say("Next: Train or resume a run -> Token folder = %s   (use the SAME tokenizer: %s)" % (target, tok_path))
    return target


def cmd(a):
    from .corpus import CorpusError
    try:
        run([x.strip() for x in a.lang.split(",") if x.strip()] if a.lang else None, a.tokenizer, a.out, a.val_every)
    except (BuildError, CorpusError) as e:
        raise SystemExit("error: %s" % e)
    except KeyboardInterrupt:
        raise SystemExit("stopped; nothing half-written was kept.")
