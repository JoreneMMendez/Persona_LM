"""Stage U4a: the Data page routes of the web UI (read only).

  GET /api/data/overview            what exists (import folder, token folders), the wizard steps with done / NEXT, the exact terminal commands
  GET /api/data/audit               the privacy audit of the Messenger import: COUNTS from stats.json only
  GET /api/data/report?folder=&size=   the data-report verdict for one token folder (same numbers as `data-report`)

Rules (tests in tests/test_ui_u4a.py):
  * No route writes anything and none starts a command. The scan and the import stay in the terminal: the raw Messenger export is
    outside the project and the UI never reads it, so the wizard shows the exact commands (with a Copy button) and then checks what
    they produced.
  * Only fixed folder names are accepted (IMPORT_DIRS, TOKEN_DIRS); the page cannot send a path. Every name still goes through PathGuard.
  * The audit reads stats.json only. It never opens private/ (real names, audit.txt), train.jsonl or val.jsonl, so no message text,
    name or phone number can reach the page. persona_settings.json is read for one yes/no ("is an export folder remembered"); its path is not sent.
"""
import json
import os

from . import report as R
from .bpe import Tokenizer
from .model import PRESETS, preset

IMPORT_DIRS = ("data_msgr",)
TOKEN_DIRS = ("data_chat", "data_chat4096", "data_pre")
TOKENIZER = "work/tok.json"
DEFAULT_SIZE = "s30m"
MAX_JSON = 2_000_000                                  # stats.json is small; refuse a huge file instead of reading it


class DataProblem(Exception):
    def __init__(self, msg, code=400):
        Exception.__init__(self, msg)
        self.code = code


class DataHub:
    def __init__(self, guard):
        self.guard = guard

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
