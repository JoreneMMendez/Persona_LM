"""The chat core of the web UI (stage U1a). One loaded AI, one conversation, one reply at a time.

Safety rules kept here (each has a test in tests/test_ui_u1a.py):
  * A message from the page goes to runtime.AI.chat(), which uses ChatSession.say(), never handle(): text such as "/approve 1",
    "/reset" or "CHANGE IDENTITY" is only a message to the AI. This file has no approve, reject, identity, memory-write or persona route.
  * What the AI writes can only appear in the page as text. Nothing here reads it as a command.
  * Files are named by paths relative to the project and go through PathGuard (inside the project, never a private folder).
  * What the owner types is never logged or printed by this file.
"""
import os
import re
import threading
import time

from . import runtime as RT

TICK_LIMIT = 4000                      # most characters in one message from the page
SYSTEM_LIMIT = 600                     # most characters in the optional system line
MODEL_TOPS = ("work", "work2", "work3")
MAX_DEPTH = 4
MAX_FOUND = 200
DEFAULTS = {"max_new": 120, "temperature": 0.5, "top_k": 40}      # 0.5: what HANDOFF.md recommends for full sentences


class ChatProblem(Exception):
    """A plain sentence for the page. code is the HTTP status to answer with."""

    def __init__(self, msg, code=400):
        Exception.__init__(self, msg)
        self.code = code


class Job:
    def __init__(self, ident, tok):
        self.id, self.tok = ident, tok
        self.stop = threading.Event()
        self.ids = []                  # new token ids so far (the reply while it grows)
        self.state = "running"         # running | done | stopped | error
        self.reply = None
        self.error = None
        self.notes = []
        self.started = time.time()
        self.ended = None

    def partial(self):
        try:
            return self.tok.decode(list(self.ids))
        except Exception:
            return ""


class ChatHub:
    def __init__(self, guard):
        self.guard = guard
        self.lock = threading.RLock()
        self.ai = None
        self.label = None
        self.job = None
        self._n = 0

    # ---------------------------------------------------------------- what can be loaded
    def _rel(self, full):
        return os.path.relpath(full, self.guard.root).replace(os.sep, "/")

    def sources(self):
        """Checkpoints, tokenizers and identity-core folders found inside the project (a read-only listing, no file is opened)."""
        ckpts, toks, cores = [], [], []
        for top in MODEL_TOPS:
            try:
                base = self.guard.resolve(top)
            except Exception:
                continue
            if not os.path.isdir(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
                depth = os.path.relpath(dirpath, base).count(os.sep) + (0 if dirpath == base else 1)
                dirnames[:] = sorted(d for d in dirnames if not d.lower().startswith("private") and d.lower() not in ("__pycache__", "_to_delete"))
                if depth >= MAX_DEPTH:
                    dirnames[:] = []
                for fn in sorted(filenames):
                    low = fn.lower()
                    full = os.path.join(dirpath, fn)
                    is_ckpt = low.startswith("ckpt") and low.endswith(".npz")
                    is_tok = low.endswith(".json") and low.startswith("tok")
                    if not (is_ckpt or is_tok):
                        continue
                    try:
                        rel = self._rel(full)
                        self.guard.resolve(rel)
                        st = os.stat(full)
                    except Exception:
                        continue
                    item = {"path": rel, "folder": self._rel(dirpath), "size_mb": round(st.st_size / 1048576, 1), "modified": int(st.st_mtime)}
                    if is_ckpt:
                        item["previous"] = ".prev." in low
                        ckpts.append(item)
                    else:
                        toks.append(item)
        for it in ckpts:                                           # the nearest tokenizer above the checkpoint is the suggested one
            folder, found = it["folder"], None
            while folder and not found:
                for t in toks:
                    if t["folder"] == folder:
                        found = t["path"]
                        break
                folder = folder.rpartition("/")[0]
            if not found:                                          # s30m runs live in work3\, the frozen tokenizer in work\
                frozen = [t["path"] for t in toks if t["path"] == "work/tok.json"]
                found = frozen[0] if frozen else (toks[0]["path"] if len(toks) == 1 else None)
            it["tokenizer"] = found
        ckpts.sort(key=lambda i: (i["previous"], -i["modified"]))
        toks.sort(key=lambda i: -i["modified"])
        try:
            top = self.guard.resolve("identity_core")
            if os.path.isfile(os.path.join(top, "core.json")):
                cores.append({"path": "identity_core"})
            elif os.path.isdir(top):
                for d in sorted(os.listdir(top)):
                    if os.path.isfile(os.path.join(top, d, "core.json")) and not d.lower().startswith("private"):
                        cores.append({"path": "identity_core/" + d})
        except Exception:
            pass
        return {"checkpoints": ckpts[:MAX_FOUND], "tokenizers": toks[:MAX_FOUND], "cores": cores[:MAX_FOUND]}

    # ---------------------------------------------------------------- loading
    def _file(self, rel, what, suffix):
        if not isinstance(rel, str) or not rel.strip():
            raise ChatProblem("Choose a %s first." % what)
        try:
            full = self.guard.resolve(rel.strip())
        except Exception as e:
            raise ChatProblem(str(e))
        if not os.path.isfile(full) or not full.lower().endswith(suffix):
            raise ChatProblem("That %s is not a %s file inside the project." % (what, suffix))
        return full

    def _say(self, e):
        return str(e).replace(self.guard.root, ".")                 # no absolute folder names in answers

    @staticmethod
    def _num(body, key, lo, hi, kind):
        if key not in body or body[key] is None:
            return DEFAULTS[key]
        v = body[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v <= hi):
            raise ChatProblem("%s must be a number from %s to %s." % (key, lo, hi))
        return kind(v)

    def load(self, body):
        with self.lock:
            if self.job is not None and self.job.state == "running":
                raise ChatProblem("The AI is still writing a reply. Press Stop first.", 409)
            kw = {"max_new": self._num(body, "max_new", 1, 1000, int), "temperature": self._num(body, "temperature", 0, 2, float),
                  "top_k": self._num(body, "top_k", 0, 1000, int), "seed": 0, "fast": True}
            system = body.get("system")
            if system is not None:
                if not isinstance(system, str) or len(system) > SYSTEM_LIMIT:
                    raise ChatProblem("The system line must be text of at most %d characters." % SYSTEM_LIMIT)
                if system.strip():
                    kw["system"] = system.strip()
            try:
                if body.get("core"):
                    rel = str(body["core"]).strip()
                    try:
                        full = self.guard.resolve(rel)
                    except Exception as e:
                        raise ChatProblem(str(e))
                    if not os.path.isfile(os.path.join(full, "core.json")):
                        raise ChatProblem("There is no identity core in that folder.")
                    version = body.get("version")
                    if version is not None and (not isinstance(version, str) or not re.match(r"^[A-Za-z0-9._-]{1,40}$", version)):
                        raise ChatProblem("The version name is 1-40 letters, digits, '.', '_' or '-'.")
                    ai, label = RT.load(full, version=version or None, **kw), rel
                else:
                    ck = self._file(body.get("ckpt"), "checkpoint", ".npz")
                    tk = self._file(body.get("tokenizer"), "tokenizer", ".json")
                    ai, label = RT.load(ckpt=ck, tokenizer=tk, **kw), self._rel(ck)
            except ChatProblem:
                raise
            except RT.RuntimeProblem as e:
                raise ChatProblem(self._say(e))
            except Exception:
                raise ChatProblem("That checkpoint and tokenizer could not be loaded together. Are they from the same model (same --size and vocabulary)?")
            self._hook(ai)
            self.ai, self.label, self.job = ai, label, None
            return self.state()

    def _hook(self, ai):
        """Lets Stop and the live reply reach the sampler. The wrapper only adds two optional arguments; the prompt and the settings are untouched."""
        orig = ai.session.generate

        def generate(ids, **kw):
            job = self.job
            if job is not None:
                kw["should_stop"] = job.stop.is_set
                kw["on_token"] = job.ids.append
            return orig(ids, **kw)
        ai.session.generate = generate

    # ---------------------------------------------------------------- state
    def tokens(self):
        s = self.ai.session
        used = len(s._message_ids("system", s.system)) if s.system else 0
        used += sum(len(s._message_ids(r, t)) for r, t in s.history)
        return {"used": used, "window": s.ctx, "prompt_budget": s.prompt_budget(), "reply_max": s.settings["max_new"]}

    def state(self):
        with self.lock:
            if self.ai is None:
                return {"loaded": False}
            s, cfg = self.ai.session, self.ai.cfg
            return {"loaded": True, "label": self.label, "name": self.ai.name, "ctx": cfg.ctx, "layers": cfg.n_layer, "d_model": cfg.d_model,
                    "settings": dict(s.settings), "has_system": bool(s.system), "tokens": self.tokens(),
                    "history": [{"role": r, "text": t} for r, t in s.history],
                    "job": self.job.id if self.job is not None and self.job.state == "running" else None}

    def count(self, text):
        with self.lock:
            if self.ai is None:
                raise ChatProblem("Load a model first.", 409)
            if not isinstance(text, str):
                raise ChatProblem("text must be a string.")
            return {"tokens": len(self.ai.session._message_ids("user", text[:TICK_LIMIT])), **self.tokens()}

    # ---------------------------------------------------------------- talking
    def send(self, message):
        with self.lock:
            if self.ai is None:
                raise ChatProblem("Load a model first.", 409)
            if self.job is not None and self.job.state == "running":
                raise ChatProblem("The AI is still writing a reply.", 409)
            if not isinstance(message, str) or not message.strip():
                raise ChatProblem("Write a message first.")
            if len(message) > TICK_LIMIT:
                raise ChatProblem("That message is too long (at most %d characters)." % TICK_LIMIT)
            self._n += 1
            job = self.job = Job("j%d" % self._n, self.ai.tok)
            threading.Thread(target=self._run, args=(job, self.ai, message), daemon=True).start()
            return {"job": job.id}

    def _run(self, job, ai, message):
        try:
            reply = ai.chat(message)                                # say() inside: the text is a message, never a command
            job.reply, job.notes = reply, list(ai.last_notes)
            job.state = "stopped" if job.stop.is_set() else "done"
        except RT.RuntimeProblem as e:
            if job.stop.is_set() and not job.ids:
                job.state = "stopped"                               # stopped before the first word: the session put everything back
            else:
                job.state, job.error = "error", self._say(e)
        except ValueError as e:
            job.state, job.error = "error", self._say(e)
        except Exception:
            job.state, job.error = "error", "Something went wrong while the AI was writing. Your message was not kept; try again."
        job.ended = time.time()

    def poll(self, ident):
        with self.lock:
            job = self.job
            if job is None or job.id != ident:
                raise ChatProblem("No such reply (the model was loaded again, or the page is old).", 404)
            out = {"state": job.state, "partial": job.partial() if job.state == "running" else (job.reply or ""), "new_tokens": len(job.ids),
                   "seconds": round((job.ended or time.time()) - job.started, 1)}
            if job.state != "running":
                out.update({"reply": job.reply, "notes": job.notes, "error": job.error, "tokens": self.tokens()})
            return out

    def stop(self):
        with self.lock:
            if self.job is not None and self.job.state == "running":
                self.job.stop.set()
                return {"stopping": True}
            return {"stopping": False}
