"""Dataset recipes: stream a public Hugging Face dataset, filter it, clean it, and turn it into pretraining text. Terminal only.

  python -m persona_lm hfdata new                    # a wizard: asks questions, PEEKS at the first rows, saves dataset_recipes/<id>.json
  python -m persona_lm hfdata list                   # every recipe and how far it got (scanned / output / tokens)
  python -m persona_lm hfdata show ID                # the recipe in plain words
  python -m persona_lm hfdata peek ID                # first ~20,000 rows: fields, sample messages, score quantiles, what each filter drops
  python -m persona_lm hfdata run ID                 # stream + filter + clean (resumable) and write the output files
  python -m persona_lm hfdata assemble ID            # build the output from the shards found so far (e.g. after Ctrl+C)
  python -m persona_lm hfdata tokens ID              # output -> token files data_pre_<id>/ with the frozen tokenizer (work\\tok.json)
  python -m persona_lm hfdata colab-kit ID           # a zip for Google Drive: package + recipe + tokenizer + notebook

The same menu item is "Dataset recipes (create / run)" in `python -m persona_lm`.

A recipe is a small JSON file (see docs/DATASET_GUIDE.md). Filters are {field, op, value}: field paths like `score`, `languages[0]`,
`stream.id`; ops from a FIXED list (==, !=, >, >=, <, <=, in, contains). Nothing is ever evaluated as code, and `trust_remote_code` is never used.

Files (all of them public text; nothing of yours is read or uploaded):
  public_text/hf_<id>/shard_000001.jsonl ...   cleaned messages, written in shards while streaming (the resume point)
  public_text/hf_<id>/state.json               rows scanned, rows kept, drop counts, which shards are finished
  public_clean/<id>/<id>.jsonl                 {"text": ...} documents, ready for `prepare --kind pretrain` (like the Language data page)
  public_clean/<id>/chat_clean.txt             the same messages, one per line, for reading
  data_pre_<id>/                               token files (never overwritten; a finished one is kept and skipped)

Safety rules (tests in tests/test_hfdata.py):
  * Every output is written to <name>.part first and moved into place only when finished. A stopped or failed step never replaces a good file.
  * Ctrl+C (or a network error) finishes the current shard, saves the state and stops. The same command continues. Streaming cannot seek, so a
    resume re-reads the dataset up to the saved row (network only, nothing is processed again) and rebuilds the duplicate list from the shards.
  * Usernames and every other column are never written: only the text field and the stream id (to group messages) are kept.
  * Changing the source, filters or cleaning of a recipe that already has shards is refused unless you add --restart (old work is MOVED to _to_delete\\).
"""
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from collections import Counter

RECIPE_DIR = "dataset_recipes"
WORK_TOP = "public_text"
CLEAN_TOP = "public_clean"
PEEK_ROWS = 20000
OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "contains")
NUM_OPS = (">", ">=", "<", "<=")
EMOTE_MODES = ("drop", "word", "keep")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,59}$")
PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*|\[\d{1,3}\])*$")
_TOK_RE = re.compile(r"[^.\[\]]+|\[\d+\]")
TAG_RE = re.compile(r"<\|([^|<>]*)\|>")
PLACEHOLDER_RE = re.compile(r"\[[A-Za-z]{2,12}\]")
MISSING = object()
SHARD_ROWS_MAX = 2_000_000          # a shard is also closed after this many scanned rows, so strict filters still save their progress

DEFAULT_CLEAN = {"min_chars": 4, "collapse_repeats": 3, "dedup": True, "emotes": "drop", "scrub": True,
                 "tags": {"liver": "[streamer]", "fans": "[fans]", "maybe-collaborator": "[streamer]", "*": ""}}


class HFError(Exception):
    pass


# ------------------------------------------------------------------ recipes
def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(r, key, lo, hi):
    v = r.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise HFError("recipe: %s must be a whole number from %d to %d (got %r)" % (key, lo, hi, v))
    return v


def _text(r, key, limit):
    v = r.get(key, "")
    if v is None:
        v = ""
    if not isinstance(v, str) or len(v) > limit:
        raise HFError("recipe: %s must be text of at most %d characters" % (key, limit))
    return v


def check_filter(f):
    if not isinstance(f, dict):
        raise HFError("recipe: a filter must be {field, op, value}")
    field, op, value = f.get("field"), f.get("op"), f.get("value")
    if not isinstance(field, str) or not PATH_RE.match(field):
        raise HFError("recipe: bad filter field %r (examples: score, languages[0], stream.id)" % (field,))
    if op not in OPS:
        raise HFError("recipe: bad filter operator %r (choose one of: %s)" % (op, " ".join(OPS)))
    if op in NUM_OPS and not _is_num(value):
        raise HFError("recipe: the filter on %s %s needs a number (got %r)" % (field, op, value))
    if op == "in" and not (isinstance(value, list) and value and all(isinstance(x, (str, int, float, bool)) for x in value)):
        raise HFError("recipe: the filter on %s in needs a list of values" % field)
    if op in ("==", "!=", "contains") and not isinstance(value, (str, int, float, bool)):
        raise HFError("recipe: the filter on %s %s needs one value (text, number, true or false)" % (field, op))
    return {"field": field, "op": op, "value": value}


def check_recipe(raw):
    """Validates a recipe (from a file or the wizard) and returns a complete copy. Raises HFError with a plain message."""
    if not isinstance(raw, dict):
        raise HFError("recipe: must be a JSON object")
    rid = raw.get("id")
    if not isinstance(rid, str) or not ID_RE.match(rid):
        raise HFError("recipe: id must be 1-40 lowercase letters, digits, - or _ (got %r)" % (rid,))
    repo = raw.get("hf_repo")
    if not isinstance(repo, str) or not REPO_RE.match(repo):
        raise HFError("recipe: hf_repo must look like owner/name (got %r). Links and paths are not accepted." % (repo,))
    r = {"id": rid, "label": _text({"label": raw.get("label") or rid}, "label", 120) or rid, "hf_repo": repo}
    for key in ("config", "revision"):
        v = raw.get(key)
        if v is not None and (not isinstance(v, str) or not NAME_RE.match(v)):
            raise HFError("recipe: %s must be a plain name or null (got %r)" % (key, v))
        r[key] = v
    split = raw.get("split", "train")
    if not isinstance(split, str) or not re.match(r"^[A-Za-z0-9_\[\]:%+-]{1,40}$", split):
        raise HFError("recipe: bad split %r" % (split,))
    r["split"] = split
    tf = raw.get("text_field")
    if not isinstance(tf, str) or not PATH_RE.match(tf):
        raise HFError("recipe: text_field must be a field name such as message or text (got %r)" % (tf,))
    r["text_field"] = tf
    filters = raw.get("filters", [])
    if not isinstance(filters, list) or len(filters) > 20:
        raise HFError("recipe: filters must be a list of at most 20 filters")
    r["filters"] = [check_filter(f) for f in filters]
    clean = raw.get("clean", {})
    if not isinstance(clean, dict):
        raise HFError("recipe: clean must be an object")
    c = json.loads(json.dumps(DEFAULT_CLEAN))
    for k, v in clean.items():
        if k not in c:
            raise HFError("recipe: unknown cleaning option %r (known: %s)" % (k, ", ".join(sorted(c))))
        c[k] = v
    c["min_chars"] = _int(c, "min_chars", 0, 10000)
    if c["collapse_repeats"] != 0:
        c["collapse_repeats"] = _int(c, "collapse_repeats", 2, 20)
    for k in ("dedup", "scrub"):
        if not isinstance(c[k], bool):
            raise HFError("recipe: clean.%s must be true or false" % k)
    if c["emotes"] not in EMOTE_MODES:
        raise HFError("recipe: clean.emotes must be one of: %s" % ", ".join(EMOTE_MODES))
    tags = c["tags"]
    if not isinstance(tags, dict) or not all(isinstance(k, str) and isinstance(v, str) and len(v) <= 30 for k, v in tags.items()):
        raise HFError("recipe: clean.tags must map a tag kind (liver, fans, *) to a short placeholder text")
    r["clean"] = c
    group = raw.get("group")
    if group is not None:
        if not isinstance(group, dict) or not isinstance(group.get("by"), str) or not PATH_RE.match(group["by"]):
            raise HFError("recipe: group must be null or {\"by\": \"stream.id\", \"lines\": 30}")
        lines = group.get("lines", 30)
        if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= 500:
            raise HFError("recipe: group.lines must be a whole number from 1 to 500")
        group = {"by": group["by"], "lines": lines}
    r["group"] = group
    r["max_kept"] = _int({"max_kept": raw.get("max_kept", 5_000_000)}, "max_kept", 1, 100_000_000)
    r["max_scanned"] = _int({"max_scanned": raw.get("max_scanned", 0)}, "max_scanned", 0, 10 ** 10)
    r["shuffle_buffer"] = _int({"shuffle_buffer": raw.get("shuffle_buffer", 0)}, "shuffle_buffer", 0, 1_000_000)
    r["shard_size"] = _int({"shard_size": raw.get("shard_size", 100_000)}, "shard_size", 100, 1_000_000)
    r["license"] = _text(raw, "license", 80)
    r["attribution"] = _text(raw, "attribution", 300)
    r["private_only"] = bool(raw.get("private_only", True))
    return r


def recipe_path(rid, root="."):
    if not isinstance(rid, str) or not ID_RE.match(rid):
        raise HFError("a recipe id is 1-40 lowercase letters, digits, - or _ (got %r)" % (rid,))
    return os.path.join(root, RECIPE_DIR, rid + ".json")


def load_recipe(rid, root="."):
    path = recipe_path(rid, root)
    if not os.path.exists(path):
        have = list_recipe_ids(root)
        raise HFError("no recipe %r in %s%s" % (rid, RECIPE_DIR, ("; you have: " + ", ".join(have)) if have else ". Make one with:  python -m persona_lm hfdata new"))
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except ValueError as e:
        raise HFError("%s is not valid JSON: %s" % (path, e))
    r = check_recipe(raw)
    if r["id"] != rid:
        raise HFError("%s: the id inside (%s) must match the file name" % (path, r["id"]))
    return r


def save_recipe(r, root="."):
    """Writes dataset_recipes/<id>.json. Never overwrites an existing recipe (raises HFError)."""
    r = check_recipe(r)
    path = recipe_path(r["id"], root)
    if os.path.exists(path):
        raise HFError("a recipe called %s already exists; choose another id" % r["id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def list_recipe_ids(root="."):
    d = os.path.join(root, RECIPE_DIR)
    if not os.path.isdir(d):
        return []
    return sorted(n[:-5] for n in os.listdir(d) if n.endswith(".json") and ID_RE.match(n[:-5]))


def work_dir(r, root="."):
    return os.path.join(root, WORK_TOP, "hf_" + r["id"])


def clean_dir(r, root="."):
    return os.path.join(root, CLEAN_TOP, r["id"])


def signature(r):
    """What the shards depend on. Raising max_kept or changing the grouping lines does not change it."""
    keys = {k: r[k] for k in ("hf_repo", "config", "revision", "split", "text_field", "filters", "clean", "shuffle_buffer")}
    keys["group_by"] = (r["group"] or {}).get("by")
    return hashlib.sha1(json.dumps(keys, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def filter_label(f):
    return "%s %s %s" % (f["field"], f["op"], json.dumps(f["value"]))


# ------------------------------------------------------------------ reading rows and filtering
def get_path(row, path):
    cur = row
    for tok in _TOK_RE.findall(path):
        if tok.startswith("["):
            i = int(tok[1:-1])
            if not isinstance(cur, (list, tuple)) or i >= len(cur):
                return MISSING
            cur = cur[i]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                return MISSING
            cur = cur[tok]
    return cur


def _eq(a, b):
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def test_filter(f, row):
    """True when the row passes. A missing field or a value of the wrong type fails (and is counted as a drop by that filter)."""
    a = get_path(row, f["field"])
    if a is MISSING:
        return False
    op, v = f["op"], f["value"]
    if op == "==":
        return _eq(a, v)
    if op == "!=":
        return not _eq(a, v)
    if op in NUM_OPS:
        if not _is_num(a):
            return False
        return {">": a > v, ">=": a >= v, "<": a < v, "<=": a <= v}[op]
    if op == "in":
        return any(_eq(a, x) for x in v)
    if isinstance(a, str):
        return isinstance(v, str) and v in a
    if isinstance(a, list):
        return any(_eq(x, v) for x in a)
    return False


def first_failing(filters, row):
    for i, f in enumerate(filters):
        if not test_filter(f, row):
            return i
    return None


class Cleaner:
    """One message -> cleaned text, or (None, reason). Counts what it replaced in `counts`."""

    def __init__(self, clean):
        self.c = clean
        n = clean["collapse_repeats"]
        self.chars = re.compile(r"(.{1,4}?)\1{%d,}" % n, re.S) if n else None
        self.words = re.compile(r"(?<!\S)(\S+)(?:\s+\1(?!\S)){%d,}" % n) if n else None
        self.n = n
        self.counts = Counter()

    def _tag(self, m):
        body = m.group(1)
        if len(body) >= 2 and body.startswith(":") and body.endswith(":"):
            self.counts["emote"] += 1
            mode = self.c["emotes"]
            if mode == "drop":
                return " "
            return " %s " % body[1:-1] if mode == "word" else " " + body + " "
        kind = body.split(":", 1)[0]
        self.counts["tag"] += 1
        tags = self.c["tags"]
        return tags.get(kind, tags.get("*", "")) or " "

    def clean(self, text):
        text = TAG_RE.sub(self._tag, text)
        text = " ".join(text.split())
        if self.chars:
            text = self.chars.sub(lambda m: m.group(1) * self.n, text)
            text = self.words.sub(lambda m: " ".join([m.group(1)] * self.n), text)
        if self.c["scrub"]:
            from .corpus import scrub_public
            text = scrub_public(text, self.counts)
        text = " ".join(text.split())
        if len(PLACEHOLDER_RE.sub("", text).strip()) < max(self.c["min_chars"], 1):
            return None, "too short"
        return text, None


def text_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def open_stream(recipe, skip=0):
    """The real dataset: an iterator of rows (dicts). Streaming only: the 30 GB are never downloaded as a whole."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise HFError("The `datasets` package is not installed. Run:  pip install datasets")
    kw = {"split": recipe["split"], "streaming": True}
    if recipe["config"]:
        kw["name"] = recipe["config"]
    if recipe["revision"]:
        kw["revision"] = recipe["revision"]
    try:
        ds = load_dataset(recipe["hf_repo"], **kw)
        if recipe["shuffle_buffer"]:
            ds = ds.shuffle(seed=0, buffer_size=recipe["shuffle_buffer"])
        if skip:
            ds = ds.skip(skip)
        return iter(ds)
    except Exception as e:
        raise HFError("could not open %s (%s: %s). Check the name, the split and your internet; a private or gated dataset needs `huggingface-cli login`."
                      % (recipe["hf_repo"], type(e).__name__, e))


# ------------------------------------------------------------------ peeking (the first rows) and describing fields
def fetch_rows(recipe, n=PEEK_ROWS, opener=None):
    it = (opener or open_stream)(recipe, 0)
    rows = []
    try:
        for row in it:
            rows.append(row)
            if len(rows) >= n:
                break
    except KeyboardInterrupt:
        raise
    except Exception as e:
        if not rows:
            raise HFError("could not read rows (%s: %s)" % (type(e).__name__, e))
    if not rows:
        raise HFError("the dataset gave no rows (wrong split or config?)")
    return rows


def _flatten(obj, prefix, seen, depth):
    if isinstance(obj, dict):
        if depth >= 3:
            return
        for k, v in obj.items():
            if isinstance(k, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", k):
                _flatten(v, (prefix + "." if prefix else "") + k, seen, depth + 1)
    elif isinstance(obj, list):
        if obj and len(obj) <= 12 and all(_is_num(x) for x in obj):
            for i, x in enumerate(obj[:3]):
                _flatten(x, "%s[%d]" % (prefix, i), seen, depth + 1)
    else:
        d = seen.setdefault(prefix, {"types": Counter(), "n": 0, "len": 0, "vals": [], "true": 0})
        d["n"] += 1
        if obj is None:
            d["types"]["null"] += 1
        elif isinstance(obj, bool):
            d["types"]["bool"] += 1
            d["true"] += int(obj)
        elif isinstance(obj, int):
            d["types"]["int"] += 1
            d["vals"].append(obj)
        elif isinstance(obj, float):
            d["types"]["float"] += 1
            d["vals"].append(obj)
        elif isinstance(obj, str):
            d["types"]["text"] += 1
            d["len"] += len(obj)


def describe_rows(rows):
    """[{path, type, n, detail, numeric, text}] for every simple field. Text examples are NOT kept (usernames stay out of the screen)."""
    seen = {}
    for row in rows:
        _flatten(row, "", seen, 0)
    out = []
    for path, d in seen.items():
        if not PATH_RE.match(path):
            continue
        kind = d["types"].most_common(1)[0][0]
        if kind in ("int", "float") and d["vals"]:
            detail = "%s .. %s" % (_short(min(d["vals"])), _short(max(d["vals"])))
        elif kind == "text":
            detail = "about %d characters" % (d["len"] // max(d["n"], 1))
        elif kind == "bool":
            detail = "%d%% true" % round(100.0 * d["true"] / max(d["n"], 1))
        else:
            detail = ""
        out.append({"path": path, "type": kind, "n": d["n"], "detail": detail, "numeric": kind in ("int", "float"), "text": kind == "text",
                    "vals": d["vals"] if kind in ("int", "float") else None, "mean_len": d["len"] / max(d["n"], 1)})
    return out


def _short(v):
    return ("%.4g" % v) if isinstance(v, float) else str(v)


def quantiles(vals, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    v = sorted(vals)
    n = len(v)
    return {q: v[min(n - 1, int(q * n))] for q in qs} if n else {}


def guess_text_field(fields):
    texts = [f for f in fields if f["text"]]
    for name in ("text", "message", "content", "body", "comment", "sentence"):
        for f in texts:
            if f["path"] == name:
                return name
    return max(texts, key=lambda f: f["mean_len"])["path"] if texts else ""


def analyze(recipe, rows, sample_n=5):
    """What the recipe would do to the sampled rows. Pure function (no network)."""
    filters = recipe["filters"]
    drops = Counter()
    passing = []
    for row in rows:
        i = first_failing(filters, row)
        if i is None:
            passing.append(row)
        else:
            drops[i] += 1
    quant = {}
    for f in filters:
        if f["op"] in NUM_OPS and f["field"] not in quant:
            others = [g for g in filters if g["field"] != f["field"]]
            vals = [get_path(r_, f["field"]) for r_ in rows if first_failing(others, r_) is None]
            quant[f["field"]] = quantiles([v for v in vals if _is_num(v)])
    cleaner = Cleaner(recipe["clean"])
    cdrops, seen, samples, kept = Counter(), set(), [], 0
    tag_rows = 0
    for row in passing:
        t = get_path(row, recipe["text_field"])
        if not isinstance(t, str):
            cdrops["no text"] += 1
            continue
        tag_rows += 1 if "<|" in t else 0
        c, why = cleaner.clean(t)
        if c is None:
            cdrops[why] += 1
        elif recipe["clean"]["dedup"] and text_hash(c) in seen:
            cdrops["duplicate"] += 1
        else:
            seen.add(text_hash(c))
            kept += 1
            if len(samples) < sample_n:
                samples.append(c)
    return {"rows": len(rows), "drops": drops, "passing": len(passing), "quant": quant, "clean_drops": cdrops, "kept": kept, "samples": samples,
            "tag_rows": tag_rows, "replaced": dict(cleaner.counts)}


def format_peek(recipe, info):
    n = info["rows"]
    lines = ["Looked at the first %d rows." % n]
    for i, f in enumerate(recipe["filters"]):
        lines.append("  filter %d: %-34s drops %6d rows (%.1f%%)" % (i + 1, filter_label(f), info["drops"].get(i, 0), 100.0 * info["drops"].get(i, 0) / max(n, 1)))
    lines.append("  rows passing every filter: %d (%.1f%%)" % (info["passing"], 100.0 * info["passing"] / max(n, 1)))
    for field, q in info["quant"].items():
        if q:
            lines.append("  %s among rows that pass the OTHER filters: %s" % (field, "  ".join("%d%%=%s" % (round(k * 100), _short(v)) for k, v in q.items())))
    if info["passing"]:
        bits = ", ".join("%s %d" % (k, v) for k, v in sorted(info["clean_drops"].items()))
        lines.append("  after cleaning, %d messages are kept%s%s" % (info["kept"], (" (dropped: %s)" % bits) if bits else "",
                                                                    ("; %d rows had tags" % info["tag_rows"]) if info["tag_rows"] else ""))
        for s in info["samples"]:
            lines.append("    sample: %s" % (s if len(s) <= 110 else s[:107] + "..."))
    return lines


# ------------------------------------------------------------------ the streaming run (shards + state = resumable)
def _load_state(w):
    p = os.path.join(w, "state.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        if isinstance(s, dict) and isinstance(s.get("shards"), list):
            return s
    except (OSError, ValueError):
        pass
    raise HFError("%s is damaged. Run again with --restart (the old work is moved to _to_delete\\)." % p)


def _save_state(w, s):
    s["updated"] = time.time()
    os.makedirs(w, exist_ok=True)
    p = os.path.join(w, "state.json")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(s, f)
    os.replace(p + ".tmp", p)


def move_aside(path, root="."):
    """Never delete: move to _to_delete\\ with a time stamp."""
    if not os.path.exists(path):
        return None
    dest_dir = os.path.join(root, "_to_delete")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "%s_%s" % (os.path.basename(path.rstrip("\\/")), time.strftime("%Y%m%d_%H%M%S")))
    shutil.move(path, dest)
    return dest


def read_shard(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t, g = json.loads(line)
            except ValueError:
                continue
            yield t, g


class Progress:
    """A one-line live status on a real terminal; a plain line every 20 seconds anywhere else (files, pipes, Colab, PERSONA_PLAIN=1)."""

    def __init__(self, goal, live=False, stream=None, clock=time.time):
        self.goal, self.live, self.out, self.clock = goal, live, stream or sys.stdout, clock
        self.t0 = self.last = clock()
        self.k0 = 0
        self.drawn = False

    def start_from(self, kept):
        self.k0 = kept

    def update(self, scanned, kept, force=False):
        now = self.clock()
        if not force and now - self.last < (0.5 if self.live else 20.0):
            return
        self.last = now
        rate = (kept - self.k0) / max(now - self.t0, 1e-6)
        frac = min(1.0, kept / max(self.goal, 1))
        eta = ((self.goal - kept) / rate) if rate > 0 and kept < self.goal else None
        bar = "[" + "#" * int(frac * 20) + "-" * (20 - int(frac * 20)) + "]"
        line = "%s %3d%%  scanned %s  kept %s / %s  %.0f kept/s  elapsed %s  ETA %s" % (
            bar, frac * 100, format(scanned, ","), format(kept, ","), format(self.goal, ","), rate, _hms(now - self.t0), _hms(eta) if eta is not None else "?")
        if self.live:
            width = shutil.get_terminal_size((100, 24)).columns - 1
            self.out.write("\r\x1b[2K" + line[:width])
            self.out.flush()
            self.drawn = True
        else:
            print(line, file=self.out, flush=True)

    def log(self, msg):
        if self.drawn:
            self.out.write("\r\x1b[2K")
            self.drawn = False
        print(msg, file=self.out, flush=True)

    def close(self):
        if self.drawn:
            self.out.write("\r\x1b[2K")
            self.out.flush()
            self.drawn = False


def _hms(seconds):
    s = int(max(0, seconds))
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def _caps_reached(r, s):
    return s["kept"] >= r["max_kept"] or bool(r["max_scanned"] and s["rows_scanned"] >= r["max_scanned"])


def scan(recipe, root=".", opener=None, say=print, progress=None, restart=False):
    """Streams the dataset, filters and cleans it into shards. Returns the state dict (state['ended'] is 'cap', 'end' or None when stopped).
    Ctrl+C and network errors finish the current shard first; Ctrl+C is re-raised after the state is saved, a network error becomes an HFError."""
    r = recipe
    w = work_dir(r, root)
    state = _load_state(w) if not restart else None
    sig = signature(r)
    if restart and os.path.exists(w):
        dest = move_aside(w, root)
        say("Old work moved to %s" % dest)
    if state and state.get("sig") != sig:
        raise HFError("the source, filters or cleaning of recipe %s changed since the last run, so the saved shards no longer match. "
                      "Run again with --restart (the old work is moved to _to_delete\\, not deleted)." % r["id"])
    if state is None:
        state = {"sig": sig, "rows_scanned": 0, "kept": 0, "shards": [], "drops": {}, "ended": None}
    if state["ended"] == "end" or (state["ended"] == "cap" and _caps_reached(r, state)) or (state["ended"] is None and _caps_reached(r, state)):
        say("Already scanned: %s messages kept from %s rows." % (format(state["kept"], ","), format(state["rows_scanned"], ",")))
        return state
    state["ended"] = None
    cleaner = Cleaner(r["clean"])
    seen = set()
    if r["clean"]["dedup"] and state["shards"]:
        say("Resuming: rebuilding the duplicate list from %d finished shards..." % len(state["shards"]))
        for name in state["shards"]:
            for t, _ in read_shard(os.path.join(w, name)):
                seen.add(text_hash(t))
    drops = Counter(state["drops"])
    os.makedirs(w, exist_ok=True)
    progress = progress or Progress(r["max_kept"])
    progress.start_from(state["kept"])
    if state["rows_scanned"]:
        say("Resuming at row %s (%s kept so far). The dataset is read again up to that row first; this only uses the network." % (
            format(state["rows_scanned"], ","), format(state["kept"], ",")))
    it = (opener or open_stream)(r, state["rows_scanned"])
    group_by = (r["group"] or {}).get("by")
    scanned, kept = state["rows_scanned"], state["kept"]
    shard_f, shard_n, shard_start_scanned = None, 0, scanned
    stopped, error, ended = False, None, None

    def finalize():
        nonlocal shard_f, shard_n, shard_start_scanned
        if shard_f is not None:
            shard_f.close()
            shard_f = None
            part = os.path.join(w, "shard_%06d.jsonl.part" % (len(state["shards"]) + 1))
            if shard_n:
                name = "shard_%06d.jsonl" % (len(state["shards"]) + 1)
                os.replace(part, os.path.join(w, name))
                state["shards"].append(name)
            else:
                try:
                    os.remove(part)
                except OSError:
                    pass
        state["rows_scanned"], state["kept"], state["drops"] = scanned, kept, dict(drops)
        _save_state(w, state)
        shard_n, shard_start_scanned = 0, scanned

    try:
        while True:
            try:
                row = next(it)
            except StopIteration:
                ended = "end"
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                error = e
                break
            scanned += 1
            i = first_failing(r["filters"], row)
            if i is not None:
                drops["filter %d: %s" % (i + 1, filter_label(r["filters"][i]))] += 1
            else:
                t = get_path(row, r["text_field"])
                if not isinstance(t, str):
                    drops["no text"] += 1
                else:
                    c, why = cleaner.clean(t)
                    if c is None:
                        drops[why] += 1
                    else:
                        h = text_hash(c) if r["clean"]["dedup"] else None
                        if h is not None and h in seen:
                            drops["duplicate"] += 1
                        else:
                            if h is not None:
                                seen.add(h)
                            if shard_f is None:
                                shard_f = open(os.path.join(w, "shard_%06d.jsonl.part" % (len(state["shards"]) + 1)), "w", encoding="utf-8")
                            g = get_path(row, group_by) if group_by else None
                            shard_f.write(json.dumps([c, g if isinstance(g, (str, int)) else None], ensure_ascii=False) + "\n")
                            shard_n += 1
                            kept += 1
            if scanned % 500 == 0:
                progress.update(scanned, kept)
            if shard_n >= r["shard_size"] or scanned - shard_start_scanned >= SHARD_ROWS_MAX:
                finalize()
            if kept >= r["max_kept"] or (r["max_scanned"] and scanned >= r["max_scanned"]):
                ended = "cap"
                break
    except KeyboardInterrupt:
        stopped = True
    progress.update(scanned, kept, force=True)
    progress.close()
    state["ended"] = ended
    finalize()
    if stopped:
        say("Stopped. %s messages kept from %s rows; the shards and the position are saved. Run the same command to continue." % (format(kept, ","), format(scanned, ",")))
        raise KeyboardInterrupt()
    if error is not None:
        raise HFError("the download stopped (%s: %s). %s messages kept so far are saved; run the same command to continue." % (
            type(error).__name__, error, format(kept, ",")))
    say("Scan finished (%s): %s rows read, %s messages kept." % ("the dataset ended" if ended == "end" else "your cap was reached", format(scanned, ","), format(kept, ",")))
    if drops:
        say("Dropped: " + "; ".join("%s: %s" % (k, format(v, ",")) for k, v in sorted(drops.items(), key=lambda kv: -kv[1])))
    return state


# ------------------------------------------------------------------ output files
def _group_docs(msgs, group):
    if not group:
        for t, _ in msgs:
            yield t, 1
        return
    buf, cur, limit = [], None, group["lines"]
    for t, g in msgs:
        if buf and (g != cur or len(buf) >= limit):
            yield "\n".join(buf), len(buf)
            buf = []
        cur = g
        buf.append(t)
    if buf:
        yield "\n".join(buf), len(buf)


def assemble(recipe, root=".", say=print, partial=False):
    """Shards -> public_clean/<id>/<id>.jsonl + chat_clean.txt. Both are written as .part files and moved in place at the end."""
    r = recipe
    w = work_dir(r, root)
    state = _load_state(w)
    if not state or not state["shards"]:
        raise HFError("nothing has been scanned yet. Run:  python -m persona_lm hfdata run %s" % r["id"])
    if state.get("ended") is None and not partial:
        raise HFError("the scan of %s is not finished (%s messages so far). Run it again to continue, or use `assemble` to build the output from what exists." % (
            r["id"], format(state["kept"], ",")))
    out_dir = clean_dir(r, root)
    os.makedirs(out_dir, exist_ok=True)
    final, chat = os.path.join(out_dir, r["id"] + ".jsonl"), os.path.join(out_dir, "chat_clean.txt")
    n_msgs, n_docs, chars, samples = 0, 0, 0, []

    def msgs():
        nonlocal n_msgs, chars
        for name in state["shards"]:
            for t, g in read_shard(os.path.join(w, name)):
                n_msgs += 1
                chars += len(t)
                if len(samples) < 5:
                    samples.append(t)
                fc.write(t + "\n")
                yield t, g

    try:
        with open(final + ".part", "w", encoding="utf-8") as fj, open(chat + ".part", "w", encoding="utf-8") as fc:
            for doc, _ in _group_docs(msgs(), r["group"]):
                fj.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
                n_docs += 1
        os.replace(final + ".part", final)
        os.replace(chat + ".part", chat)
    finally:
        for p in (final + ".part", chat + ".part"):
            try:
                os.remove(p)
            except OSError:
                pass
    say("Output: %s messages in %s documents, %.1f MB of text" % (format(n_msgs, ","), format(n_docs, ","), chars / 1e6))
    say("  %s\n  %s" % (final, chat))
    if r["license"] or r["attribution"]:
        say("  Source: %s | licence: %s | %s" % (r["hf_repo"], r["license"] or "unknown", r["attribution"] or "credit the dataset authors"))
    return {"messages": n_msgs, "documents": n_docs, "chars": chars, "samples": samples, "path": final}


def make_tokens(recipe, root=".", tokenizer=None, out=None, also=(), val_every=50, new=False, say=print):
    """Output -> token files with the frozen tokenizer (only READ). The folder appears only when finished; an existing finished one is kept."""
    from . import buildtokens as B
    from . import data as D
    from .bpe import Tokenizer
    r = recipe
    src = os.path.join(clean_dir(r, root), r["id"] + ".jsonl")
    if not os.path.exists(src):
        raise HFError("no output yet (%s is missing). Run:  python -m persona_lm hfdata run %s" % (src, r["id"]))
    for p in also:
        if not os.path.exists(p):
            raise HFError("--also: %s does not exist" % p)
    try:
        tok_path = B.find_tokenizer(tokenizer)
    except B.BuildError as e:
        raise HFError(str(e))
    target = out or os.path.join(root, "data_pre_" + r["id"])
    if os.path.exists(os.path.join(target, "meta.json")) and not new:
        say("%s is already built (it has meta.json); nothing to do. Use --new for another folder." % target)
        return target
    if new:
        target = B.new_folder(target)
    elif os.path.isdir(target) and os.listdir(target):
        raise HFError("%s already has files in it. Pick another --out, or use --new." % target)
    part = target + ".part"
    shutil.rmtree(part, ignore_errors=True)
    say("Tokenizer: %s (read only). Making tokens into %s ..." % (tok_path, target))
    files = [src] + list(also)
    t0 = time.time()
    try:
        tok = Tokenizer.load(tok_path)
        D.prepare(tok, D.pretrain_docs(files, tok), part, with_mask=False, val_every=val_every, log=lambda m: say("  " + m))
        if not os.path.exists(os.path.join(part, "train.bin")) or os.path.getsize(os.path.join(part, "train.bin")) == 0:
            raise HFError("no tokens were produced (empty train.bin).")
        if os.path.isdir(target):
            os.rmdir(target)
        os.replace(part, target)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    n_train = os.path.getsize(os.path.join(target, "train.bin")) // 2
    n_val = os.path.getsize(os.path.join(target, "val.bin")) // 2
    say("DONE in %d s: %s (train %s tokens, validation %s tokens)" % (time.time() - t0, target, format(n_train, ","), format(n_val, ",")))
    say("Next: python -m persona_lm data-report --data-dir %s --size s30m --tokenizer %s" % (target, tok_path))
    return target


# ------------------------------------------------------------------ Colab kit
def colab_kit(recipe, root=".", say=print):
    """A zip to unpack into MyDrive: persona/persona_lm, persona/dataset_recipes/<id>.json, persona/tok.json (if found), the notebook, a README."""
    from . import buildtokens as B
    r = recipe
    nb = os.path.join(root, "colab_hfdata.ipynb")
    if not os.path.exists(nb):
        raise HFError("colab_hfdata.ipynb is missing from the project folder")
    pkg = os.path.dirname(os.path.abspath(__file__))
    n = 1
    zpath = os.path.join(root, "colab_kit_%s.zip" % r["id"])
    while os.path.exists(zpath):
        n += 1
        zpath = os.path.join(root, "colab_kit_%s_%d.zip" % (r["id"], n))
    with open(nb, encoding="utf-8") as f:
        notebook = json.load(f)
    for cell in notebook.get("cells", []):                     # put this recipe's id into the settings cell
        src = cell.get("source", [])
        for i, line in enumerate(src):
            if re.match(r'^RECIPE_ID\s*=\s*"', line):
                src[i] = re.sub(r'"[^"]*"', '"%s"' % r["id"], line, count=1)
    text = json.dumps(notebook, indent=1, ensure_ascii=False)
    try:
        tok = B.find_tokenizer(None)
    except B.BuildError:
        tok = None
    part = zpath + ".part"
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
            for base, dirs, names in os.walk(pkg):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for name in names:
                    if name.endswith((".pyc", ".pyo")):
                        continue
                    full = os.path.join(base, name)
                    z.write(full, os.path.join("persona", "persona_lm", os.path.relpath(full, pkg)))
            z.write(recipe_path(r["id"], root), "persona/dataset_recipes/%s.json" % r["id"])
            if tok:
                z.write(tok, "persona/tok.json")
            z.writestr("persona/colab_hfdata.ipynb", text)
            z.writestr("persona/README.txt", _KIT_README % {"id": r["id"], "tok": "included (work\\tok.json, only read)" if tok else "NOT found: copy work\\tok.json to MyDrive/persona/tok.json yourself"})
        os.replace(part, zpath)
    finally:
        try:
            os.remove(part)
        except OSError:
            pass
    say("Wrote %s (%.1f MB). Unzip it into Google Drive's MyDrive (it makes MyDrive/persona/), then open persona/colab_hfdata.ipynb in Colab." % (zpath, os.path.getsize(zpath) / 1e6))
    say("To continue from your pretrained s30m, also copy work3\\run_pre\\ckpt.npz to MyDrive/persona/base_ckpt.npz (public text only).")
    return zpath


_KIT_README = """Colab kit for recipe %(id)s

1. Unzip this into your Google Drive MyDrive, so you get MyDrive/persona/ with persona_lm/, dataset_recipes/, colab_hfdata.ipynb.
2. Tokenizer: %(tok)s
3. Optional: copy work3\\run_pre\\ckpt.npz to MyDrive/persona/base_ckpt.npz to continue from your pretrained s30m.
4. In Colab: Runtime > Change runtime type > T4 GPU, open persona/colab_hfdata.ipynb from Drive, run the cells from the top.
   After a disconnect just run all the cells again: it continues by itself (shards and checkpoints live in Drive).
Only public text goes through this. Nothing of your chats is in the kit.
"""


# ------------------------------------------------------------------ the wizard
def _q(ask, say, label, default="", check=None, hint=""):
    """One question. Returns the answer (the default on Enter). `check(value)` returns an error text or None."""
    while True:
        note = "[%s] " % default if default != "" else ""
        val = ask("  %s %s%s> " % (label, ("(%s) " % hint) if hint else "", note)).strip().strip('"').strip()
        val = val or default
        if check:
            problem = check(val)
            if problem:
                say("    " + problem)
                continue
        return val


def _yes(val):
    return str(val).strip().lower() in ("y", "yes", "true", "1")


def parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    return s or "dataset"


def wizard(root=".", ask=input, say=print, opener=None, rows_n=PEEK_ROWS):
    """Asks questions, shows the first rows, and saves dataset_recipes/<id>.json. Returns the recipe id, or None if the person backed out."""
    say("\n  New dataset recipe. It streams a PUBLIC Hugging Face dataset (nothing is downloaded as a whole), so you can repeat it for any dataset.")
    say("  Press Enter to accept the value in [brackets]. Ctrl+C backs out without saving.\n")
    try:
        repo = _q(ask, say, "Hugging Face dataset", "", lambda v: None if REPO_RE.match(v) else "type it as owner/name, for example pykeio/oshichats-v2 (no links)")
        split = _q(ask, say, "Split", "train", lambda v: None if re.match(r"^[A-Za-z0-9_\[\]:%+-]{1,40}$", v) else "a split name such as train")
        config = _q(ask, say, "Config / subset", "", lambda v: None if not v or NAME_RE.match(v) else "a plain name, or empty for none", hint="optional")
        probe = {"id": "probe", "hf_repo": repo, "config": config or None, "split": split, "text_field": "text"}
        say("\n  Reading the first %s rows by streaming (usually under a minute)..." % format(rows_n, ","))
        rows = fetch_rows(check_recipe(probe), rows_n, opener)
        fields = describe_rows(rows)
        say("\n  Fields found (examples of text are not shown, so usernames stay hidden):")
        for i, f in enumerate(fields, 1):
            say("   %2d  %-22s %-6s %s" % (i, f["path"], f["type"], f["detail"]))
        by_num = {str(i): f["path"] for i, f in enumerate(fields, 1)}
        text_paths = {f["path"] for f in fields if f["text"]}
        guess = guess_text_field(fields)

        def pick(v):
            return by_num.get(v, v)

        text_field = pick(_q(ask, say, "Which field is the message text? (name or number)", guess,
                             lambda v: None if pick(v) in text_paths else "choose one of the text fields: " + ", ".join(sorted(text_paths))))
        numeric = [f for f in fields if f["numeric"] and f["vals"]]
        if numeric:
            say("\n  Number fields, to help you choose thresholds (10%, 50%, 90% of the sampled rows):")
            for f in numeric[:10]:
                q = quantiles(f["vals"], (0.1, 0.5, 0.9))
                say("   %-22s %s" % (f["path"], "  ".join("%d%%=%s" % (round(k * 100), _short(v)) for k, v in q.items())))
        filters = []
        say("\n  Filters keep only the rows you want. Add them one at a time; press Enter on the field to finish.")
        while len(filters) < 20:
            field = _q(ask, say, "Filter %d: field (name or number, Enter = done)" % (len(filters) + 1), "",
                       lambda v: None if not v or (pick(v) in {f["path"] for f in fields}) else "not a field of this dataset (see the list above)")
            if not field:
                break
            field = pick(field)
            op = _q(ask, say, "  operator", ">" if any(f["path"] == field and f["numeric"] for f in fields) else "==",
                    lambda v: None if v in OPS else "choose one of: " + " ".join(OPS))
            value = parse_value(_q(ask, say, "  value", "", lambda v: None if v else "a value is needed (a number, true, false, or text)", hint="number, true/false or text"))
            try:
                flt = check_filter({"field": field, "op": op, "value": value})
            except HFError as e:
                say("    " + str(e))
                continue
            filters.append(flt)
            alone = sum(1 for row in rows if test_filter(flt, row))
            every = sum(1 for row in rows if first_failing(filters, row) is None)
            say("    this filter alone keeps %.1f%% of the sampled rows; all filters together keep %.1f%%." % (100.0 * alone / len(rows), 100.0 * every / len(rows)))
        clean = {"min_chars": int(_q(ask, say, "Drop messages shorter than N characters", "4", lambda v: None if v.isdigit() and int(v) <= 10000 else "a whole number")),
                 "collapse_repeats": int(_q(ask, say, "Collapse repeated characters/words to at most N copies (0 = off)", "3",
                                            lambda v: None if v.isdigit() and (v == "0" or 2 <= int(v) <= 20) else "0, or a number from 2 to 20")),
                 "dedup": _yes(_q(ask, say, "Remove exact duplicate messages? (y/n)", "y")),
                 "emotes": _q(ask, say, "Emotes like <|:name:|>: drop, word or keep", "drop", lambda v: None if v in EMOTE_MODES else "drop, word or keep")}
        group = None
        by = _q(ask, say, "Group consecutive messages that share a field into one document? (field, Enter = no)", "",
                lambda v: None if not v or pick(v) in {f["path"] for f in fields} else "not a field of this dataset")
        if by:
            lines = int(_q(ask, say, "  at most how many messages per document", "30", lambda v: None if v.isdigit() and 1 <= int(v) <= 500 else "1 to 500"))
            group = {"by": pick(by), "lines": lines}
        max_kept = int(_q(ask, say, "Stop after keeping how many messages", "5000000", lambda v: None if v.isdigit() and int(v) >= 1 else "a whole number"))
        default_id = slug(repo.split("/")[-1])
        existing = set(list_recipe_ids(root))
        rid = _q(ask, say, "Short id for this recipe", default_id,
                 lambda v: "use 1-40 lowercase letters, digits, - or _" if not ID_RE.match(v) else ("a recipe called %s exists; pick another" % v if v in existing else None))
        label = _q(ask, say, "Label", repo)
        lic = _q(ask, say, "Licence (see the dataset card; Enter = unknown)", "", lambda v: None if len(v) <= 80 else "too long")
        attribution = _q(ask, say, "Attribution / credit line", repo, lambda v: None if len(v) <= 300 else "too long")
        recipe = check_recipe({"id": rid, "label": label, "hf_repo": repo, "config": config or None, "split": split, "text_field": text_field, "filters": filters,
                               "clean": clean, "group": group, "max_kept": max_kept, "license": lic, "attribution": attribution, "private_only": True})
        say("\n  Preview on the sampled rows:")
        for line in format_peek(recipe, analyze(recipe, rows)):
            say("  " + line)
        if not _yes(_q(ask, say, "Save this recipe? (y/n)", "y")):
            say("  not saved.")
            return None
        path = save_recipe(recipe, root)
    except (KeyboardInterrupt, EOFError):
        say("\n  backed out; nothing was saved.")
        return None
    say("\n  Saved %s" % path)
    say("  Next:  python -m persona_lm hfdata run %s      (then: hfdata tokens %s)" % (recipe["id"], recipe["id"]))
    return recipe["id"]


# ------------------------------------------------------------------ commands
def describe_recipe(r):
    lines = ["Recipe %s: %s" % (r["id"], r["label"]), "  source   : %s (split %s%s)" % (r["hf_repo"], r["split"], ", config " + r["config"] if r["config"] else ""),
             "  text     : field %s" % r["text_field"]]
    lines += ["  filter   : " + filter_label(f) for f in r["filters"]] or []
    c = r["clean"]
    lines.append("  cleaning : min %d chars, collapse repeats to %s, %s, emotes %s, links/e-mails/@handles replaced: %s" % (
        c["min_chars"], c["collapse_repeats"] or "off", "remove duplicates" if c["dedup"] else "keep duplicates", c["emotes"], "yes" if c["scrub"] else "no"))
    lines.append("  keep     : up to %s messages%s%s" % (format(r["max_kept"], ","), (", read at most %s rows" % format(r["max_scanned"], ",")) if r["max_scanned"] else "",
                                                       (", shuffle buffer %d" % r["shuffle_buffer"]) if r["shuffle_buffer"] else ""))
    lines.append("  documents: %s" % (("%d messages of the same %s per document" % (r["group"]["lines"], r["group"]["by"])) if r["group"] else "one message per document"))
    lines.append("  licence  : %s%s" % (r["license"] or "unknown", " (private training only)" if r["private_only"] else ""))
    return lines


def status(r, root="."):
    s = None
    try:
        s = _load_state(work_dir(r, root))
    except HFError:
        pass
    out = os.path.join(clean_dir(r, root), r["id"] + ".jsonl")
    meta = os.path.join(root, "data_pre_" + r["id"], "meta.json")
    parts = []
    if s:
        parts.append("scanned %s rows, kept %s%s" % (format(s["rows_scanned"], ","), format(s["kept"], ","), "" if s.get("ended") else " (unfinished)"))
    else:
        parts.append("not scanned")
    parts.append("output %.0f MB" % (os.path.getsize(out) / 1e6) if os.path.exists(out) else "no output")
    if os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8") as f:
                parts.append("tokens %s" % format(json.load(f).get("train_tokens", 0), ","))
        except (OSError, ValueError):
            parts.append("tokens ?")
    else:
        parts.append("no tokens")
    return "; ".join(parts)


def _live_wanted(plain):
    if plain or os.environ.get("PERSONA_PLAIN") or os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(sys.stdout, "isatty", None)
    if not (isatty and isatty()):
        return False
    from .pretty import enable_vt
    return enable_vt()


def confirm_run(r, ask=input, say=print):
    """The box shown before any download: what, how big, licence, privacy. Returns True to start."""
    mb = r["max_kept"] * 70 * 3 / 1e6
    say("\n  About to stream %s from Hugging Face (public data; nothing of yours is read or uploaded)." % r["hf_repo"])
    for line in describe_recipe(r)[2:]:
        say("  " + line)
    say("  Disk: about %d MB for shards + output + chat_clean.txt. Network: up to the whole file if the filters keep few rows." % mb)
    if r["private_only"]:
        say("  Licence: %s. Treated as private training only: do not re-upload the cleaned text or publish a model without checking the licence." % (r["license"] or "unknown"))
    say("  Ctrl+C is safe: it saves the position, and the same command continues.")
    return _yes(ask("  Start? (y/N) > "))


def resolve_id(a, root="."):
    if a.recipe:
        return a.recipe
    ids = list_recipe_ids(root)
    if len(ids) == 1:
        return ids[0]
    raise HFError("which recipe? %s" % (("choose one of: " + ", ".join(ids)) if ids else "there are none yet; make one with:  python -m persona_lm hfdata new"))


ACTIONS = ("new", "list", "show", "peek", "run", "assemble", "tokens", "colab-kit")


def add_arguments(p):
    p.add_argument("action", choices=ACTIONS, help="new = wizard; list; show; peek = look at the first rows; run = stream + clean + write; "
                   "assemble = build the output from the shards so far; tokens = make token files; colab-kit = a zip for Google Drive")
    p.add_argument("recipe", nargs="?", default=None, help="the recipe id (optional when there is only one)")
    p.add_argument("--plain", action="store_true", help="plain log lines instead of the live status line")
    p.add_argument("--yes", action="store_true", help="run: skip the 'Start?' question")
    p.add_argument("--restart", action="store_true", help="run: start the scan again (old shards are MOVED to _to_delete\\)")
    p.add_argument("--rows", type=int, default=PEEK_ROWS, help="peek: how many rows to look at (default %d)" % PEEK_ROWS)
    p.add_argument("--max-kept", type=int, default=None, help="run: override the recipe's maximum number of messages for this run")
    p.add_argument("--tokenizer", default=None, help="tokens: default work\\tok.json, else tok.json (only read)")
    p.add_argument("--out", default=None, help="tokens: output folder (default data_pre_<id>)")
    p.add_argument("--also", nargs="*", default=[], help="tokens: more text files/folders to add, e.g. public_mix\\tl.jsonl public_mix\\ceb.jsonl (keeps other languages in the mix)")
    p.add_argument("--val-every", type=int, default=50, help="tokens: every Nth document goes to validation")
    p.add_argument("--new", action="store_true", help="tokens: make a NEW folder (_2, _3, ...) even if one is finished")


def cmd(a, root=".", ask=input, say=print, opener=None):
    try:
        if a.action == "new":
            wizard(root, ask, say, opener)
            return
        if a.action == "list":
            ids = list_recipe_ids(root)
            if not ids:
                say("No recipes yet. Make one with:  python -m persona_lm hfdata new")
            for rid in ids:
                try:
                    r = load_recipe(rid, root)
                    say("  %-16s %-34s %s" % (rid, r["hf_repo"], status(r, root)))
                except HFError as e:
                    say("  %-16s PROBLEM: %s" % (rid, e))
            return
        r = load_recipe(resolve_id(a, root), root)
        if a.action == "show":
            for line in describe_recipe(r):
                say(line)
            say("  status   : " + status(r, root))
        elif a.action == "peek":
            say("Reading the first %s rows of %s by streaming..." % (format(a.rows, ","), r["hf_repo"]))
            rows = fetch_rows(r, a.rows, opener)
            if not r["filters"]:
                say("This recipe has no filters yet.")
            for line in format_peek(r, analyze(r, rows)):
                say(line)
        elif a.action == "run":
            if a.max_kept:
                r = check_recipe(dict(r, max_kept=a.max_kept))
            if not a.yes and not confirm_run(r, ask, say):
                say("not started.")
                return
            progress = Progress(r["max_kept"], live=_live_wanted(a.plain))
            state = scan(r, root, opener, say=progress.log, progress=progress, restart=a.restart)
            if state["ended"]:
                info = assemble(r, root, say=say)
                for s in info["samples"][:3]:
                    say("  sample: %s" % (s if len(s) <= 110 else s[:107] + "..."))
                say("Next:  python -m persona_lm hfdata tokens %s" % r["id"])
        elif a.action == "assemble":
            assemble(r, root, say=say, partial=True)
        elif a.action == "tokens":
            make_tokens(r, root, a.tokenizer, a.out, a.also, a.val_every, a.new, say=say)
        elif a.action == "colab-kit":
            colab_kit(r, root, say=say)
    except HFError as e:
        raise SystemExit("error: %s" % e)
    except KeyboardInterrupt:
        raise SystemExit("stopped; what was written is kept. Run the same command to continue.")
