"""Language data: download public text (English, Tagalog, Cebuano, Taglish), clean it, optionally build the pretraining mix, check it.

  python -m persona_lm corpus                          # all four languages: download, clean, check
  python -m persona_lm corpus --lang tl,taglish        # only some
  python -m persona_lm corpus --mode clean             # skip the download: clean what is already in public_text/
  python -m persona_lm corpus --mix                    # also rebuild public_mix/ (balanced files for `prepare`)
  python -m persona_lm corpus --mb en=200 tl=150       # size caps in MB (0 = all; Cebuano has no cap)

The web UI's "Language data" page starts exactly this command in the background (persona_lm/ui_corpus.py).

It does NOT reimplement anything: each step calls the existing scripts in the project folder
  English  grab_english.stream_into  -> public_clean/en/en.jsonl            (FineWeb-Edu is already cleaned)
  Tagalog  grab_data.grab            -> public_text/tl/tl.jsonl      -> clean_corpus.clean_tl -> public_clean/tl/tl.jsonl
  Cebuano  ceb_domains.main          -> public_text/ceb/ceb_url.jsonl -> clean_ceb.main      -> public_clean/ceb/ceb.jsonl
  Taglish  grab_taglish.main         -> public_text/taglish/pex.jsonl -> clean_taglish (here) -> public_clean/taglish/taglish.jsonl
  --mix    make_mix.build            -> public_mix/<lang>.jsonl
so the page, this command and running the scripts by hand give the same files.

Safety rules (tests in tests/test_ui_corpus.py):
  * Every output is written to <name>.part first and moved into place only when its step finished. A stopped or failed step never
    replaces a good existing file with a half-written one.
  * Stop (Ctrl+C, or the page's Stop button) ends after the current document; nothing after it runs.
  * Taglish text (PinoyExchange forum posts) has e-mails, links, @handles and phone numbers replaced before it reaches public_clean/.
  * Only public text goes through here. Nothing of yours is read, and nothing is uploaded.
"""
import argparse
import importlib
import json
import os
import shutil
import sys
import time
from collections import Counter

LANG_ORDER = ("en", "tl", "ceb", "taglish")

# id -> what the page and the log show. Paths are relative to the project folder.
LANGS = {
    "en": {"label": "English", "source": "FineWeb-Edu (sample-10BT)", "license": "ODC-By", "raw": None, "clean": "public_clean/en/en.jsonl",
           "default_mb": 420, "capped": True, "mix_mb": 330,
           "note": "Already cleaned and de-duplicated by its publisher; documents under 300 characters are dropped while downloading."},
    "tl": {"label": "Tagalog", "source": "FineWeb-2 (fil_Latn)", "license": "ODC-By", "raw": "public_text/tl/tl.jsonl", "clean": "public_clean/tl/tl.jsonl",
           "default_mb": 300, "capped": True, "mix_mb": 195,
           "note": "Web text. Cleaning drops short pages and pages whose first line repeats (website boilerplate)."},
    "ceb": {"label": "Cebuano", "source": "FineWeb-2 (ceb_Latn)", "license": "ODC-By", "raw": "public_text/ceb/ceb_url.jsonl", "clean": "public_clean/ceb/ceb.jsonl",
            "default_mb": 0, "capped": False, "mix_mb": 65,
            "note": "Downloads all of it (it is small) with each page's website, then keeps only Philippine and hand-picked sites: most web Cebuano is machine-translated."},
    "taglish": {"label": "Taglish", "source": "PinoyExchange forum threads (gabtan99/pex-conversations)", "license": "unknown (private training only)",
                "raw": "public_text/taglish/pex.jsonl", "clean": "public_clean/taglish/taglish.jsonl", "default_mb": 100, "capped": True, "mix_mb": 60,
                "note": "Real forum users wrote these posts. For your own private training only: do not re-upload the data or publish a model trained on it "
                        "without checking the site's terms. E-mails, links, @handles and phone numbers are replaced during cleaning."},
}
MODES = ("all", "clean")
MAX_MB = 5000
EN_SHORT_FRACTION = 0.9          # an English download that ends below this share of what was asked is treated as cut off (network), not finished
TAGLISH_MIN_CHARS = 200
MIN_CHARS = 300
MAX_SAME_OPENING = 3


class CorpusError(Exception):
    pass


# ------------------------------------------------------------------ planning (also used by the web UI to show the steps)
def parse_mb(items):
    """['en=200', 'tl=150'] -> {'en': 200, 'tl': 150}. Raises CorpusError on anything else."""
    out = {}
    for it in items or []:
        k, sep, v = str(it).partition("=")
        if not sep or k not in LANGS:
            raise CorpusError("--mb wants language=MB with language one of %s (got %r)" % (", ".join(LANG_ORDER), it))
        try:
            n = int(v)
        except ValueError:
            raise CorpusError("the MB for %s must be a whole number (got %r)" % (k, v))
        if not 0 <= n <= MAX_MB:
            raise CorpusError("the MB for %s must be between 0 and %d" % (k, MAX_MB))
        out[k] = n
    return out


def check_langs(langs):
    asked = set(langs or [])
    if not asked or asked - set(LANG_ORDER):
        raise CorpusError("choose at least one of: %s" % ", ".join(LANG_ORDER))
    return [l for l in LANG_ORDER if l in asked]


def plan_stages(langs, mode="all", mix=False):
    """The ordered steps: [{'kind': 'download'|'clean'|'mix'|'check', 'lang': id or None, 'title': text}]."""
    langs = check_langs(langs)
    if mode not in MODES:
        raise CorpusError("mode must be one of: %s" % ", ".join(MODES))
    stages = []
    for l in langs:
        label = LANGS[l]["label"]
        if mode == "all":
            stages.append({"kind": "download", "lang": l, "title": "Download %s (%s)" % (label, LANGS[l]["source"])})
        stages.append({"kind": "clean", "lang": l, "title": ("Check %s (already cleaned)" if l == "en" else "Clean %s") % label})
    if mix:
        stages.append({"kind": "mix", "lang": None, "title": "Build the training mix (public_mix/)"})
    stages.append({"kind": "check", "lang": None, "title": "Check that every file can be read by `prepare`"})
    return stages


# ------------------------------------------------------------------ small helpers
def _say(msg):
    print(msg, flush=True)


def _mb(path):
    try:
        return os.path.getsize(path) / 1e6
    except OSError:
        return 0.0


STAGE_HOOK = None          # set by corpus_ui (the terminal progress screen): called as hook(index, total, title, lang) when a step starts


def _write_progress(path, index, total, title, lang=None):
    if STAGE_HOOK is not None:
        try:
            STAGE_HOOK(index, total, title, lang)
        except Exception:                          # a display problem must never stop a download
            pass
    if not path:
        return
    rec = {"index": index, "total": total, "title": title, "lang": lang, "at": time.time()}
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, path)
    except OSError:
        pass


def _script(name):
    """Imports one of the project's root scripts (grab_data, clean_ceb, ...). The project folder is the current folder."""
    if sys.path[:1] != [os.getcwd()]:                 # the project's own scripts win over any other copy
        sys.path.insert(0, os.getcwd())
        importlib.invalidate_caches()
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == "datasets":
            raise CorpusError("The `datasets` package is not installed. Run:  pip install datasets")
        raise CorpusError("%s.py was not found in the project folder (%s). Run this from the project folder." % (name, os.getcwd()))


def _drop(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _promote(part, final):
    os.makedirs(os.path.dirname(final) or ".", exist_ok=True)
    os.replace(part, final)


class _Stop:
    def __init__(self, stopfile):
        self.stopfile = stopfile

    def asked(self):
        return bool(self.stopfile and os.path.exists(self.stopfile))

    def check(self):
        if self.asked():
            raise KeyboardInterrupt()


# ------------------------------------------------------------------ taglish cleaning (new: the other languages reuse the old scripts)
def scrub_public(text, counts=None):
    """Replace e-mails, links, @handles and phone numbers in public forum text. Uses the same patterns as the Messenger anonymiser."""
    from .anonymize import DATE_RE, EMAIL_RE, HANDLE_RE, PHONE_CANDIDATE_RE, URL_RE
    counts = counts if counts is not None else Counter()

    def sub(rx, repl, kind, s):
        def f(m):
            counts[kind] += 1
            return repl
        return rx.sub(f, s)

    def phone(m):
        s = m.group(0)
        digits = sum(c.isdigit() for c in s)
        if DATE_RE.match(s.strip()) or not 7 <= digits <= 15:
            return s
        counts["phone"] += 1
        return "[PHONE]"

    text = sub(URL_RE, "[URL]", "url", text)
    text = sub(EMAIL_RE, "[EMAIL]", "email", text)
    text = sub(HANDLE_RE, "[HANDLE]", "handle", text)
    return PHONE_CANDIDATE_RE.sub(phone, text)


def clean_taglish(src, dst):
    """Forum threads -> clean lines. Scrub, drop short threads, exact duplicates and boilerplate openings (more than MAX_SAME_OPENING)."""
    openings = Counter()
    with open(src, encoding="utf-8") as f:
        for line in f:
            try:
                openings[json.loads(line)["text"][:80]] += 1
            except (ValueError, KeyError, TypeError):
                continue
    drop, scrubbed, seen, kept, chars = Counter(), Counter(), set(), 0, 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            try:
                text = json.loads(line)["text"]
            except (ValueError, KeyError, TypeError):
                drop["unreadable line"] += 1
                continue
            if not isinstance(text, str) or len(text) < TAGLISH_MIN_CHARS:
                drop["too short"] += 1
            elif openings[text[:80]] > MAX_SAME_OPENING:
                drop["repeated opening"] += 1
            else:
                text = scrub_public(text, scrubbed)
                h = hash(text)
                if h in seen:
                    drop["duplicate"] += 1
                    continue
                seen.add(h)
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                kept += 1
                chars += len(text)
    _say("Taglish: kept %d threads, %.0f MB. Dropped: %s. Replaced: %s" % (kept, chars / 1e6, dict(drop), dict(scrubbed)))
    if kept == 0:
        raise CorpusError("Taglish cleaning kept nothing. The raw file may be empty or in a different format.")


# ------------------------------------------------------------------ one step each
def _need_new_taglish(mod):
    """The old grab_taglish.py streamed through `datasets` and sat at 0% for up to an hour. Never run it again."""
    try:
        with open(mod.__file__, encoding="utf-8") as f:
            old = "load_dataset" in f.read()
    except OSError:
        old = False
    if old:
        raise CorpusError("grab_taglish.py in the project folder is the OLD version (it stalls at 0%). "
                          "Copy the new grab_taglish.py into the project folder, then press Start again.")


def _download(lang, mb, stop):
    spec = LANGS[lang]
    cap = (mb if mb is not None else spec["default_mb"]) * 1_000_000
    if lang == "en":
        final = spec["clean"]
        part = final + ".part"
        mod = _script("grab_english")
        os.makedirs(os.path.dirname(final), exist_ok=True)
        want = cap or spec["default_mb"] * 1_000_000
        with open(part, "w", encoding="utf-8") as f:
            docs, chars = mod.stream_into(f, "HuggingFaceFW/fineweb-edu", "sample-10BT", want, "FineWeb-Edu")
        stop.check()
        if chars < want * EN_SHORT_FRACTION:
            raise CorpusError("English download ended early (%.0f of %.0f MB; probably the network). The partial file is kept as %s; "
                              "the earlier public_clean/en/en.jsonl was not touched. Press Start again." % (chars / 1e6, want / 1e6, part))
        _promote(part, final)
        _say("English: %d documents, %.0f MB -> %s" % (docs, chars / 1e6, final))
        return
    raw = spec["raw"]
    part = raw + ".part"
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    if lang == "tl":
        mod = _script("grab_data")
        mod.grab("fil_Latn", part, cap or 10 ** 12)
    elif lang == "ceb":
        mod = _script("ceb_domains")
        mod.OUT = part
        mod.main()
    else:
        mod = _script("grab_taglish")
        _need_new_taglish(mod)
        mod.OUT_PATH = part
        mod.main(int(cap / 1_000_000) if cap else 0)
    stop.check()
    if not os.path.exists(part) or os.path.getsize(part) == 0:
        raise CorpusError("%s: nothing was downloaded." % spec["label"])
    _promote(part, raw)
    _say("%s: %.0f MB downloaded -> %s" % (spec["label"], _mb(raw), raw))


def _clean(lang, stop):
    spec = LANGS[lang]
    if lang == "en":
        if not os.path.exists(spec["clean"]):
            raise CorpusError("English has not been downloaded yet (public_clean/en/en.jsonl is missing).")
        _say("English: %.0f MB, already clean (FineWeb-Edu). Nothing to do." % _mb(spec["clean"]))
        return
    raw, final = spec["raw"], spec["clean"]
    if not os.path.exists(raw):
        raise CorpusError("%s has not been downloaded yet (%s is missing). Run it with mode = download + clean." % (spec["label"], raw))
    part = final + ".part"
    os.makedirs(os.path.dirname(final), exist_ok=True)
    try:
        if lang == "tl":
            _script("clean_corpus").clean_tl(raw, part)
        elif lang == "ceb":
            mod = _script("clean_ceb")
            mod.SRC, mod.DST = raw, part
            mod.main()
        else:
            clean_taglish(raw, part)
        stop.check()
        if not os.path.exists(part) or os.path.getsize(part) == 0:
            raise CorpusError("%s cleaning kept nothing, so the old file was left as it is." % spec["label"])
        _promote(part, final)
    finally:
        _drop(part)


def _mix(langs, stop):
    mod = _script("make_mix")
    out_dir = "public_mix"
    tmp_dir = os.path.join(out_dir, ".part")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    mod.OUT_DIR = tmp_dir
    made = []
    try:
        for l in langs:
            spec = LANGS[l]
            if not os.path.exists(spec["clean"]):
                raise CorpusError("%s is not cleaned yet, so it cannot go into the mix." % spec["label"])
            stop.check()
            mod.build(l, spec["clean"], int(spec["mix_mb"] * 1_000_000))
            made.append(l)
        stop.check()
        os.makedirs(out_dir, exist_ok=True)
        for l in made:
            os.replace(os.path.join(tmp_dir, l + ".jsonl"), os.path.join(out_dir, l + ".jsonl"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def read_check(path, stop=None):
    """(documents, characters, bad lines) of one {"text": ...} per line file. Never keeps the text."""
    docs = chars = bad = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if stop is not None and i % 2000 == 0:
                stop.check()
            if not line.strip():
                continue
            try:
                t = json.loads(line)["text"]
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if not isinstance(t, str) or not t.strip():
                bad += 1
                continue
            docs += 1
            chars += len(t)
    return docs, chars, bad


def _check(langs, mix, stop):
    problems = []
    for l in langs:
        path = LANGS[l]["clean"]
        if not os.path.exists(path):
            problems.append("%s: %s is missing" % (LANGS[l]["label"], path))
            continue
        docs, chars, bad = read_check(path, stop)
        _say("READY  %-8s %8d documents  %7.1f MB  %s" % (LANGS[l]["label"], docs, chars / 1e6, path))
        if bad:
            problems.append("%s: %d unreadable lines in %s" % (LANGS[l]["label"], bad, path))
        if docs == 0:
            problems.append("%s: no documents in %s" % (LANGS[l]["label"], path))
        if mix:
            mp = "public_mix/%s.jsonl" % l
            if os.path.exists(mp):
                d2, c2, b2 = read_check(mp, stop)
                _say("MIX    %-8s %8d documents  %7.1f MB  %s" % (LANGS[l]["label"], d2, c2 / 1e6, mp))
                if b2 or d2 == 0:
                    problems.append("%s: the mix file %s is not readable" % (LANGS[l]["label"], mp))
    if problems:
        raise CorpusError("Check failed: " + "; ".join(problems))
    _say("All files are readable. Next: the Training page (or `pipeline plan`) turns them into token files.")


# ------------------------------------------------------------------ the whole run
def run(langs, mode="all", mb=None, mix=False, stopfile=None, progress=None, root=None):
    """Runs every stage in order. Returns normally when everything finished; raises KeyboardInterrupt when stopped, CorpusError on a problem."""
    if root:
        os.chdir(root)
    mb = dict(mb or {})
    stages = plan_stages(langs, mode, mix)
    langs = check_langs(langs)
    stop = _Stop(stopfile)
    total = len(stages)
    _say("Language data: %s | mode: %s | mix: %s | %d steps" % (", ".join(LANGS[l]["label"] for l in langs), mode, "yes" if mix else "no", total))
    for i, st in enumerate(stages, 1):
        stop.check()
        _write_progress(progress, i, total, st["title"], st["lang"])
        _say("[STEP %d/%d] %s" % (i, total, st["title"]))
        t0 = time.time()
        if st["kind"] == "download":
            _download(st["lang"], mb.get(st["lang"]), stop)
        elif st["kind"] == "clean":
            _clean(st["lang"], stop)
        elif st["kind"] == "mix":
            _mix(langs, stop)
        else:
            _check(langs, mix, stop)
        _say("   done in %d s" % (time.time() - t0))
    _write_progress(progress, total, total, "Finished", None)
    _say("FINISHED")


# ------------------------------------------------------------------ command line glue (persona_lm.cli calls these)
def add_arguments(p):
    p.add_argument("--lang", default=",".join(LANG_ORDER), help="comma list of: %s (default: all four)" % ", ".join(LANG_ORDER))
    p.add_argument("--mode", choices=MODES, default="all", help="all = download then clean (default); clean = only clean what is already in public_text/")
    p.add_argument("--mb", nargs="*", default=[], metavar="LANG=MB", help="size caps, e.g. en=200 tl=150 taglish=0 (0 = all). Cebuano always downloads all.")
    p.add_argument("--mix", action="store_true", help="also rebuild public_mix/ for the chosen languages (what `make_mix.py` does)")
    p.add_argument("--plain", action="store_true", help="plain log lines instead of the live progress screen (the screen only appears on a real terminal anyway)")
    p.add_argument("--stopfile", default=None, help=argparse.SUPPRESS)       # used by the web UI: the file whose existence means 'stop'
    p.add_argument("--progress", default=None, help=argparse.SUPPRESS)       # used by the web UI: where to write the step number


def cmd(a):
    try:
        if not a.plain and not a.stopfile and not a.progress:           # the web UI passes stopfile/progress: it always gets plain lines
            from . import corpus_ui
            if corpus_ui.wanted():
                corpus_ui.run_with_display([x.strip() for x in a.lang.split(",") if x.strip()], mode=a.mode, mb=parse_mb(a.mb), mix=a.mix)
                return
        run([x.strip() for x in a.lang.split(",") if x.strip()], mode=a.mode, mb=parse_mb(a.mb), mix=a.mix, stopfile=a.stopfile, progress=a.progress)
    except CorpusError as e:
        raise SystemExit("error: %s" % e)
