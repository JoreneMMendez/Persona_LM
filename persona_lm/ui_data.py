"""Stage U4a: the Data page routes of the web UI (read only).

  GET /api/data/overview            what exists (import folder, token folders), the wizard steps with done / NEXT, the exact terminal commands
  GET /api/data/audit               the privacy audit of the Messenger import: COUNTS from stats.json only
  GET /api/data/report?folder=&size=   the data-report verdict for one token folder (same numbers as `data-report`)
  GET /api/data/clean-preview?max_dup=&max_dup_long=&keep_others=   U4b: what clean_chat.py WOULD do to data_msgr (in memory, writes nothing)
  GET /api/data/repeats?...&text=1  U4b: the most repeated replies (text only when asked, only if the import was anonymised, cut to 30 characters)
  POST /api/data/clean-apply {plan_id}   U4b: write data_msgr_chats for the previewed plan (owner's button; old files are MOVED to _to_delete\\)

Rules (tests in tests/test_ui_u4a.py):
  * U4a routes write nothing and none starts a command. U4b adds ONE write, clean-apply: it writes only data_msgr_chats (the folder clean_chat.py
    writes), only for a plan that was previewed on files that have not changed since, and it never deletes: an existing data_msgr_chats is moved
    to _to_delete\\. It runs clean_chat.py's own functions, so the page and `python clean_chat.py --dir data_msgr` give the same files.
  * Counts only, except the opt-in repeats?text=1 (see above). Reading train.jsonl for the cleaner happens inside the server; no text is sent otherwise.
  * (U4a) the scan and import stay in the terminal. The scan and the import stay in the terminal: the raw Messenger export is
    outside the project and the UI never reads it, so the wizard shows the exact commands (with a Copy button) and then checks what
    they produced.
  * Only fixed folder names are accepted (IMPORT_DIRS, TOKEN_DIRS); the page cannot send a path. Every name still goes through PathGuard.
  * The audit reads stats.json only. It never opens private/ (real names, audit.txt), train.jsonl or val.jsonl, so no message text,
    name or phone number can reach the page. persona_settings.json is read for one yes/no ("is an export folder remembered"); its path is not sent.
"""
import collections
import hashlib
import importlib.util
import json
import os
import threading
import time

from . import report as R
from .bpe import Tokenizer
from .model import PRESETS, preset

IMPORT_DIRS = ("data_msgr",)
TOKEN_DIRS = ("data_chat", "data_chat4096", "data_pre")
TOKENIZER = "work/tok.json"
DEFAULT_SIZE = "s30m"
MAX_JSONL = 400_000_000                               # largest train.jsonl / val.jsonl the cleaner will read
MAX_DUP_LIMIT = 100
REPEAT_TEXT_ROWS = 15
REPEAT_TEXT_CHARS = 30
CLEAN_OUT = "data_msgr_chats"
MAX_JSON = 2_000_000                                  # stats.json is small; refuse a huge file instead of reading it


class DataProblem(Exception):
    def __init__(self, msg, code=400):
        Exception.__init__(self, msg)
        self.code = code


class DataHub:
    def __init__(self, guard):
        self.guard = guard
        self.lock = threading.Lock()
        self.plans = {}                                   # plan_id -> {"params": ..., "sig": ...}, kept in RAM only

    def _clean(self, e):
        return str(e).replace(self.guard.root, ".")

    def _path(self, name):
        try:
            return self.guard.resolve(name)
        except Exception as e:
            raise DataProblem(self._clean(e), 403)

    def _json(self, path):
        try:
            if os.path.getsize(path) > MAX_JSON:
                return None
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else None
        except (OSError, ValueError):
            return None

    # ---------------------------------------------------------------- small facts
    def _export_remembered(self):
        try:
            path = self._path("persona_settings.json")
        except DataProblem:
            return False
        if not os.path.exists(path):
            return False
        return bool((self._json(path) or {}).get("export"))

    def _import_info(self, name):
        try:
            folder = self._path(name)
        except DataProblem:
            return {"name": name, "exists": False}
        stats_path = os.path.join(folder, "stats.json")
        if not os.path.isdir(folder):
            return {"name": name, "exists": False}
        out = {"name": name, "exists": True, "has_stats": os.path.exists(stats_path),
               "train_jsonl": os.path.exists(os.path.join(folder, "train.jsonl")), "val_jsonl": os.path.exists(os.path.join(folder, "val.jsonl"))}
        s = self._json(stats_path) if out["has_stats"] else None
        if s:
            out.update({"chats": s.get("chats_imported"), "sessions": s.get("sessions"), "my_messages": s.get("my_messages"),
                        "examples_train": s.get("examples_train"), "examples_val": s.get("examples_val"), "period": s.get("period"),
                        "learn_from_others": s.get("learn_from_others"), "anonymized": s.get("anonymized")})
        return out

    def _token_info(self, name):
        try:
            folder = self._path(name)
        except DataProblem:
            return {"name": name, "exists": False}
        meta, train = os.path.join(folder, "meta.json"), os.path.join(folder, "train.bin")
        if not (os.path.exists(meta) and os.path.exists(train)):
            return {"name": name, "exists": False}
        m = self._json(meta) or {}
        val = os.path.join(folder, "val.bin")
        return {"name": name, "exists": True, "kind": m.get("kind", "?"), "train_tokens": os.path.getsize(train) // 2,
                "val_tokens": os.path.getsize(val) // 2 if os.path.exists(val) else 0}

    # ---------------------------------------------------------------- overview + wizard
    def overview(self):
        imports = [self._import_info(n) for n in IMPORT_DIRS]
        tokens = [self._token_info(n) for n in TOKEN_DIRS]
        remembered = self._export_remembered()
        imp = imports[0]
        chat_tokens = [t for t in tokens if t["exists"] and t["kind"] == "chat"]
        steps = [
            {"id": "scan", "title": "1. Scan your Messenger export",
             "what": "Lists the chats in your Facebook export and finds your name. Read only. It runs in the terminal because the export is outside this project folder and the UI never reads it.",
             "command": 'python -m persona_lm messenger-scan "C:\\path\\to\\your_facebook_activity"',
             "done": remembered, "done_text": "An export folder is remembered (settings file)."},
            {"id": "import", "title": "2. Import and anonymise the chats",
             "what": "Turns the chats into train.jsonl and val.jsonl. Names, numbers, e-mails and links are replaced, and a self-audit counts anything that slipped through. It learns from everyone's messages and all chats; add --only-mine to restrict.",
             "command": "python -m persona_lm messenger-import --out data_msgr",
             "done": bool(imp.get("has_stats") and imp.get("train_jsonl")), "done_text": "data_msgr has train.jsonl and stats.json."},
            {"id": "audit", "title": "3. Check the privacy audit (below)",
             "what": "Read the audit on this page before you do anything else with the data. Zero leaks is what you want. Suspicious items are reviewed in the terminal, never on this page.",
             "command": None, "done": False, "done_text": "", "manual": True},
            {"id": "prepare", "title": "4. Turn the chats into token files",
             "what": "Uses the frozen tokenizer. Chat tokens go to data_chat (or data_chat4096 for the s30m window).",
             "command": "python -m persona_lm prepare --kind chat --tokenizer work\\tok.json --data data_msgr\\train.jsonl --val-data data_msgr\\val.jsonl --out data_chat",
             "done": bool(chat_tokens), "done_text": "A chat token folder exists."},
            {"id": "report", "title": "5. Read the data-report verdict (below)",
             "what": "Says whether the amount of chat text is enough for the model size, and whether the tokenizer matches.",
             "command": None, "done": False, "done_text": "", "manual": True},
        ]
        nxt = next((s["id"] for s in steps if not s["done"] and not s.get("manual")), None)
        for s in steps:
            s["state"] = "done" if s["done"] else ("next" if s["id"] == nxt else "todo")
        return {"imports": imports, "token_folders": tokens, "steps": steps, "export_remembered": remembered,
                "sizes": sorted(PRESETS), "default_size": DEFAULT_SIZE,
                "note": "Private files (data_msgr/private/, raw exports, train.jsonl, val.jsonl) are never opened by this page."}

    # ---------------------------------------------------------------- privacy audit (stats.json only)
    def audit(self, name="data_msgr"):
        if name not in IMPORT_DIRS:
            raise DataProblem("unknown import folder")
        folder = self._path(name)
        stats_path = os.path.join(folder, "stats.json")
        if not os.path.exists(stats_path):
            return {"exists": False, "folder": name}
        s = self._json(stats_path)
        if s is None:
            raise DataProblem("stats.json could not be read (damaged or too big). Run messenger-import again.", 422)
        audit = s.get("audit") or {}
        leaks = {str(k): int(v) for k, v in (audit.get("leaks") or {}).items() if isinstance(v, (int, float))}
        susp = {str(k): int(v) for k, v in (audit.get("suspicious") or {}).items() if isinstance(v, (int, float))}
        counts = {str(k): int(v) for k, v in (s.get("anonymizer_counts") or {}).items() if isinstance(v, (int, float))}
        anonymized = bool(s.get("anonymized"))
        n_leaks = sum(leaks.values())
        if not anonymized:
            level, text = "bad", "Anonymisation was OFF for this import. Use this data only on your own computer and never upload it."
        elif n_leaks:
            level, text = "bad", "%d possible leak(s) found by the self-audit. Do not upload this data. Open data_msgr\\private\\audit.txt in a text editor (the UI will not show it), fix the names, and import again." % n_leaks
        elif susp:
            level, text = "warn", "No leaks, but the audit marked %d thing(s) to look at. Check data_msgr\\private\\capitalised_words.txt for names of people who are not in the chats." % sum(susp.values())
        else:
            level, text = "ok", "The self-audit found no leaks and nothing suspicious."
        return {"exists": True, "folder": name, "anonymized": anonymized, "paranoid": bool(s.get("paranoid")),
                "safe_to_upload": bool(s.get("safe_to_upload")), "level": level, "verdict": text,
                "leaks": leaks, "suspicious": susp, "masked": counts,
                "dropped_message_kinds": {str(k): int(v) for k, v in (s.get("dropped_message_kinds") or {}).items() if isinstance(v, (int, float))},
                "size_level": s.get("data_size_level"), "size_verdict": s.get("data_size_verdict"),
                "my_messages": s.get("my_messages"), "chats": s.get("chats_imported"), "sessions": s.get("sessions"),
                "period": s.get("period"), "learn_from_others": s.get("learn_from_others"),
                "note": "Counts only. No message text, name or number is shown here."}

    # ---------------------------------------------------------------- data-report verdict
    def report(self, folder, size):
        if folder not in TOKEN_DIRS:
            raise DataProblem("unknown token folder")
        if size not in PRESETS:
            raise DataProblem("unknown model size")
        path = self._path(folder)
        try:
            s = R.data_summary(path)
        except FileNotFoundError:
            raise DataProblem("%s has no token files yet. Run the prepare command first." % folder, 404)
        except (OSError, ValueError) as e:
            raise DataProblem(self._clean(e), 422)
        cfg = preset(size, s["vocab_size"])
        params = cfg.n_params()
        loss = s["loss_tokens"]
        measure = loss if loss is not None else s["train_tokens"]
        chat = loss is not None
        if chat:
            verdict = R._chat_verdict(loss)
            level = "bad" if loss < 50_000 else ("warn" if loss < 500_000 else "ok")
        else:
            verdict = R._pretrain_verdict(measure / params)
            tpp = measure / params
            level = "bad" if tpp < 1 else ("warn" if tpp < 20 else "ok")
        warnings = []
        if s["val_tokens"] <= 2 * cfg.ctx:
            warnings.append("Validation data is smaller than 2 windows of %d tokens, so validation loss will NOT be computed." % cfg.ctx)
        tok = {"state": "none", "text": "No tokenizer was checked (%s not found)." % TOKENIZER}
        try:
            tpath = self._path(TOKENIZER)
            if os.path.exists(tpath):
                got = Tokenizer.load(tpath).fingerprint()
                if not s["tokenizer_sha256"]:
                    tok = {"state": "unknown", "text": "These files were made before fingerprints existed, so the tokenizer cannot be verified. Run prepare again to be safe."}
                elif got == s["tokenizer_sha256"]:
                    tok = {"state": "ok", "text": "The tokenizer matches (%s)." % TOKENIZER}
                else:
                    tok = {"state": "bad", "text": "DIFFERENT tokenizer from %s. Do not train with this data and that tokenizer; run prepare again." % TOKENIZER}
        except DataProblem:
            pass
        except Exception:
            tok = {"state": "unknown", "text": "The tokenizer file could not be read."}
        passes = None
        if chat and s["train_tokens"]:
            per_step = 1 * 4 * cfg.ctx                    # batch 1 x accum 4, the setting that fits the 6 GB GPU
            passes = {"tokens_per_step": per_step, "steps_for_3_passes": int(round(3 * s["train_tokens"] / per_step)),
                      "note": "HANDOFF: this model overfits after about 3 passes over the chat data, so stop near that many steps."}
        return {"folder": folder, "size": size, "kind": s["kind"], "train_tokens": s["train_tokens"], "val_tokens": s["val_tokens"],
                "examples": s["documents"], "loss_tokens": loss, "params": params, "window": cfg.ctx, "vocab": cfg.vocab_size,
                "tokens_per_param": round(measure / params, 3), "level": level, "verdict": verdict, "warnings": warnings,
                "tokenizer": tok, "passes": passes}

    # ================================================================ U4b: the cleaner
    def _cleaner(self):
        """clean_chat.py from the project folder (fixed file name), loaded as a module. Its functions are the ones the terminal command uses."""
        path = os.path.join(self.guard.root, "clean_chat.py")
        if not os.path.isfile(path):
            raise DataProblem("clean_chat.py is not in the project folder.", 404)
        spec = importlib.util.spec_from_file_location("plm_clean_chat", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            raise DataProblem("clean_chat.py could not be loaded: " + self._clean(e), 500)
        for need in ("load_dir", "run_clean", "write_output"):
            if not hasattr(mod, need):
                raise DataProblem("clean_chat.py is an older version (no %s). Copy the newest one from the project zip." % need, 409)
        return mod

    @staticmethod
    def _int(v, default, name):
        try:
            n = default if v in (None, "") else int(v)
        except (TypeError, ValueError):
            raise DataProblem("%s must be a whole number." % name)
        if n < 0 or n > MAX_DUP_LIMIT:
            raise DataProblem("%s must be between 0 and %d." % (name, MAX_DUP_LIMIT))
        return n

    def _params(self, src):
        keep = src.get("keep_others")
        return {"max_dup": self._int(src.get("max_dup"), 3, "max_dup"), "max_dup_long": self._int(src.get("max_dup_long"), 5, "max_dup_long"),
                "keep_others": keep in (True, "1", 1)}

    def _inputs(self):
        folder = self._path("data_msgr")
        train, val = os.path.join(folder, "train.jsonl"), os.path.join(folder, "val.jsonl")
        if not os.path.exists(train):
            raise DataProblem("No import yet: data_msgr\\train.jsonl is missing. Do wizard steps 1 and 2 first.", 404)
        sig = []
        for p in (train, val):
            if os.path.exists(p):
                st = os.stat(p)
                if st.st_size > MAX_JSONL:
                    raise DataProblem("%s is too big for the page (over %d MB). Run clean_chat.py in the terminal." % (os.path.basename(p), MAX_JSONL // 10 ** 6), 413)
                sig.append([st.st_size, st.st_mtime_ns])
            else:
                sig.append(None)
        return folder, sig

    def _run(self, params):
        folder, sig = self._inputs()
        mod = self._cleaner()
        try:
            raw_train, raw_val = mod.load_dir(folder)
            raw_targets = self._targets(raw_train)
            cleaned, stats = mod.run_clean(raw_train, raw_val, params["keep_others"], params["max_dup"], params["max_dup_long"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise DataProblem("The chat files could not be read (%s). Run messenger-import again." % type(e).__name__, 422)
        return mod, sig, raw_train, raw_targets, cleaned, stats

    @staticmethod
    def _targets(bucket):
        """Reply targets (assistant, loss on) of a list of (order, chat, examples), normalised the way clean_chat.py counts them."""
        out = []
        for _, _, exs in bucket:
            for e in exs:
                t = [m for m in e.get("messages", []) if m.get("train")]
                if t and t[-1].get("role") == "assistant":
                    out.append(str(t[-1].get("content", "")).strip().lower())
        return out

    @staticmethod
    def _buckets(counter):
        edges = [("2-3 times", 2, 3), ("4-10 times", 4, 10), ("11-50 times", 11, 50), ("over 50 times", 51, 10 ** 9)]
        rows = []
        for label, lo, hi in edges:
            vals = [n for n in counter.values() if lo <= n <= hi]
            rows.append({"label": label, "replies": len(vals), "examples": sum(vals)})
        return rows

    def _reply_summary(self, targets):
        c = collections.Counter(targets)
        total = len(targets)
        top10 = sum(n for _, n in c.most_common(10))
        return {"total": total, "distinct": len(c), "top10_share": round(top10 / total, 4) if total else 0.0, "buckets": self._buckets(c)}

    def _plan_id(self, params, sig):
        raw = json.dumps([params, sig], sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def clean_preview(self, src):
        params = self._params(src)
        with self.lock:
            mod, sig, raw_train, raw_targets, cleaned, stats = self._run(params)
            kept = [str(m["content"]).strip().lower() for exs in cleaned["train"].values() for e in exs for m in e["messages"] if m.get("train") and m.get("role") == "assistant"]
            splits = {}
            for sp in ("train", "val"):
                s = stats[sp]
                splits[sp] = {"before": s["examples_in"], "after": s["examples_out"], "removed": {k: s[k] for k in (
                    "system_msgs_removed", "dropped_no_target", "dropped_other_group_speaker", "dropped_repeated_short", "dropped_repeated_long")}}
            pid = self._plan_id(params, sig)
            if len(self.plans) > 20:
                self.plans.clear()
            self.plans[pid] = {"params": params, "sig": sig}
            out_info = self._out_info()
            kw = " --keep-others" if params["keep_others"] else ""
            return {"params": params, "plan_id": pid, "splits": splits,
                    "chats": {"before": len(raw_train), "with_train_file": len(cleaned["train"]), "val_only": len([k for k in cleaned["val"] if k not in cleaned["train"]])},
                    "replies": {"before": self._reply_summary(raw_targets), "after": self._reply_summary(kept)},
                    "output": out_info,
                    "command": "python clean_chat.py --dir data_msgr --max-dup %d --max-dup-long %d%s" % (params["max_dup"], params["max_dup_long"], kw),
                    "note": "Preview only: nothing was written."}

    def _out_info(self):
        try:
            out = self._path(CLEAN_OUT)
        except DataProblem:
            return {"folder": CLEAN_OUT, "usable": False, "exists": False, "train_files": 0}
        tr = os.path.join(out, "train")
        n = len([f for f in os.listdir(tr) if f.endswith(".jsonl")]) if os.path.isdir(tr) else 0
        return {"folder": CLEAN_OUT, "usable": True, "exists": os.path.isdir(out), "train_files": n}

    def repeats(self, src, text=False):
        params = self._params(src)
        if text:
            stats = self._json(os.path.join(self._path("data_msgr"), "stats.json")) or {}
            if not stats.get("anonymized"):
                raise DataProblem("This import was not anonymised, so the page will not show its text. Look at the file yourself.", 403)
        with self.lock:
            mod, sig, raw_train, raw_targets, cleaned, stats = self._run(params)
            kept = [str(m["content"]).strip().lower() for exs in cleaned["train"].values() for e in exs for m in e["messages"] if m.get("train") and m.get("role") == "assistant"]
            before, after = collections.Counter(raw_targets), collections.Counter(kept)
            out = {"params": params, "before": self._reply_summary(raw_targets), "after": self._reply_summary(kept), "text_shown": bool(text)}
            if text:
                rows = []
                for reply, n in before.most_common(REPEAT_TEXT_ROWS):
                    if n < 2:
                        break
                    shown = reply if len(reply) <= REPEAT_TEXT_CHARS else reply[:REPEAT_TEXT_CHARS] + "\u2026"
                    rows.append({"reply": shown, "before": n, "after": after.get(reply, 0)})
                out["top"] = rows
                out["note"] = "Text from your own (anonymised) chats, cut to %d characters, at most %d rows. Shown only because you asked." % (REPEAT_TEXT_CHARS, REPEAT_TEXT_ROWS)
            return out

    def clean_apply(self, body):
        pid = body.get("plan_id")
        if not isinstance(pid, str) or pid not in self.plans:
            raise DataProblem("Preview first: there is no such plan (the server may have restarted).", 409)
        plan = self.plans[pid]
        with self.lock:
            mod, sig, raw_train, raw_targets, cleaned, stats = self._run(plan["params"])
            if sig != plan["sig"]:
                self.plans.pop(pid, None)
                raise DataProblem("The import files changed since the preview. Preview again.", 409)
            out = self._path(CLEAN_OUT)
            moved = None
            if os.path.isdir(out) and os.listdir(out):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                trash = os.path.join(self.guard.root, "_to_delete")
                os.makedirs(trash, exist_ok=True)
                dest = os.path.join(trash, "%s_%s" % (CLEAN_OUT, stamp))
                n = 1
                while os.path.exists(dest):
                    n += 1
                    dest = os.path.join(trash, "%s_%s_%d" % (CLEAN_OUT, stamp, n))
                os.rename(out, dest)
                moved = "_to_delete\\" + os.path.basename(dest)
            try:
                index, lost_val = mod.write_output(out, cleaned)
            except OSError as e:
                raise DataProblem("Could not write %s: %s" % (CLEAN_OUT, self._clean(e)), 500)
            self.plans.pop(pid, None)
        return {"written": CLEAN_OUT, "chat_files": len(index), "train_examples": sum(i["train_examples"] for i in index),
                "val_examples": sum(i["val_examples"] for i in index), "val_chats_without_train_file": len(lost_val), "moved_old_to": moved,
                "next_command": "python -m persona_lm prepare --kind chat --tokenizer work\\tok.json --data data_msgr_chats\\train --val-data data_msgr_chats\\val --out data_chat"}
