"""The chat core of the web UI (stages U1a, U1b). One loaded AI, one conversation, one reply at a time.
U1b adds: settings (temperature, top-k, longest reply), clear the conversation, save it to saved_chats/, and the mood badge.

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

from . import identity as ID
from . import personas as PERS
from . import runtime as RT
from .lineage import LineageProblem
from .bpe import ASSISTANT, END, MEMORY, MOOD, PERSONA, SPEAKER, SYSTEM, USER

TICK_LIMIT = 4000                      # most characters in one message from the page
SYSTEM_LIMIT = 600                     # most characters in the optional system line
MODEL_TOPS = ("work", "work2", "work3")
MAX_DEPTH = 4
MAX_FOUND = 200
PANEL_TEXT = 1500                      # most characters of one prompt section sent to the Prompt panel
BLOCKS = {SYSTEM: ("system", "System line"), PERSONA: ("persona", "Persona (mask)"), SPEAKER: ("person", "Person card (who you are)"),
          MOOD: ("mood", "Mood line"), MEMORY: ("memory", "Memory and old-chat recall"), USER: ("user", "You said"), ASSISTANT: ("assistant", "AI said")}
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
        self.last_prompt = None
        self._n = 0

    # ---------------------------------------------------------------- what can be loaded
    def _rel(self, full):
        return os.path.relpath(full, self.guard.root).replace(os.sep, "/")

    def sources(self):
        """Checkpoints, tokenizers and identity-core folders found inside the project (a read-only listing, no file is opened)."""
        ckpts, toks = [], []
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
        cores = self._cores()
        return {"checkpoints": ckpts[:MAX_FOUND], "tokenizers": toks[:MAX_FOUND], "cores": cores}

    def _cores(self):
        cores = []
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
        return cores[:MAX_FOUND]

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
                    if body.get("mood") is True:                    # opt-in: the typed message moves the mood, which is saved in mood.json
                        kw["mood"] = self.guard.resolve("mood.json")
                    for key, name in (("memory", "memory.json"), ("archive", "chat_archive.jsonl")):
                        if body.get(key) is True:                   # opt-in, like --memory / --archive in the terminal
                            full = self.guard.resolve(name)
                            if key == "memory" and not os.path.isfile(full):
                                raise ChatProblem("There is no memory.json in the project yet (add a memory on the Memory page first).")
                            kw[key] = full
                    ai, label = RT.load(ckpt=ck, tokenizer=tk, **kw), self._rel(ck)
            except ChatProblem:
                raise
            except RT.RuntimeProblem as e:
                raise ChatProblem(self._say(e))
            except Exception:
                raise ChatProblem("That checkpoint and tokenizer could not be loaded together. Are they from the same model (same --size and vocabulary)?")
            self._hook(ai)
            self.ai, self.label, self.job, self.last_prompt = ai, label, None, None
            return self.state()

    def _hook(self, ai):
        """Lets Stop and the live reply reach the sampler. The wrapper only adds two optional arguments; the prompt and the settings are untouched."""
        orig = ai.session.generate

        def generate(ids, **kw):
            job = self.job
            s = ai.session
            self.last_prompt = {"ids": list(ids), "memory": list(s.last_recalled), "archive": list(s.last_archive_recalled), "unfit": getattr(s, "last_unfit", 0), "history": len(s.history),
                                "budget": s.prompt_budget(), "when": time.time(), "settings": dict(kw)}      # exactly what the model is about to read
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

    def mood(self):
        m = self.ai.session.mood if self.ai is not None else None
        return None if m is None else {"label": m.label, "valence": round(m.valence, 2), "arousal": round(m.arousal, 2)}

    def state(self):
        with self.lock:
            if self.ai is None:
                return {"loaded": False}
            s, cfg = self.ai.session, self.ai.cfg
            return {"loaded": True, "label": self.label, "name": self.ai.name, "ctx": cfg.ctx, "layers": cfg.n_layer, "d_model": cfg.d_model,
                    "settings": dict(s.settings), "max_new_limit": cfg.ctx - 8, "has_system": bool(s.system), "tokens": self.tokens(), "mood": self.mood(),
                    "history": [{"role": r, "text": t} for r, t in s.history], "memory_on": s.memory is not None, "archive_on": s.archive is not None,
                    "persona": self.ai.persona, "personas": self.ai.personas.names() if self.ai.personas is not None else None,
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
                out.update({"reply": job.reply, "notes": job.notes, "error": job.error, "tokens": self.tokens(), "mood": self.mood()})
            return out

    # ---------------------------------------------------------------- U3b: personas (masks)
    def _persona_view(self, ps, rel, switchable):
        items = []
        for n in ps.names():
            p = ps.d["personas"][n]
            items.append({"name": n, "text": p["text"], "active": n == ps.active, "examples": len(p.get("examples") or []),
                          "version": p.get("version"), "stale": bool(p.get("version")) and ps.stale(n)})
        return {"loaded": True, "core": rel, "ai_name": ps.core.sections["Name"], "switchable": switchable, "active": ps.active,
                "personas": items, "block": ps.prompt_text(), "cores": self._cores()}

    def personas(self, core=None):
        """Read only. The masks of the loaded identity core (switchable), or of another core folder named by `core` (listing only)."""
        with self.lock:
            cores = self._cores()
            ai_ps = self.ai.personas if self.ai is not None else None
            if core is None or core == "":
                if ai_ps is None:
                    return {"loaded": False, "cores": cores, "has_model": self.ai is not None}
                return self._persona_view(ai_ps, self.label, True)
            if not isinstance(core, str):
                raise ChatProblem("core must be text.")
            try:
                full = self.guard.resolve(core.strip())
            except Exception as e:
                raise ChatProblem(self._say(e))
            if not os.path.isfile(os.path.join(full, "core.json")):
                raise ChatProblem("There is no identity core in that folder.")
            if ai_ps is not None and self.label == core.strip():
                return self._persona_view(ai_ps, self.label, True)
            try:
                return self._persona_view(PERS.Personas(ID.Core(full)), core.strip(), False)
            except (PERS.PersonaProblem, ID.IdentityProblem) as e:
                raise ChatProblem(self._say(e))

    def use_persona(self, name):
        """Owner-side mask switch through the runtime (the same as the typed /persona NAME). Only the page's own button reaches this: the chat route
        uses say(), so what the AI writes or what you type in the chat box can never switch a mask. The conversation is kept; only prompt text changes."""
        with self.lock:
            self._idle()
            if self.ai.personas is None:
                raise ChatProblem("This AI has no personas. Load an identity core in Chat first.", 409)
            if not isinstance(name, str) or not name.strip() or len(name) > 80:
                raise ChatProblem("Choose a persona (or none).")
            try:
                msg = self.ai.switch_persona(name)
            except RT.RuntimeProblem as e:
                raise ChatProblem(self._say(e))
            except (PERS.PersonaProblem, ID.IdentityProblem, LineageProblem, OSError) as e:
                raise ChatProblem(self._say(e) + " Nothing was changed.")
            return {"message": msg, **self.personas(), "chat": self.state()}

    # ---------------------------------------------------------------- U3a: what went into the last prompt
    @staticmethod
    def _cut(text, n=PANEL_TEXT):
        return text if len(text) <= n else text[:n].rstrip() + " [...]"

    def prompt(self):
        """Read-only. The exact token ids the model was given for the newest message, cut into their blocks, plus which memories and old-chat
        exchanges were recalled. Nothing here changes the AI, the memory or the archive."""
        with self.lock:
            if self.ai is None:
                raise ChatProblem("Load a model first.", 409)
            lp = self.last_prompt
            s = self.ai.session
            if lp is None:
                return {"has_prompt": False, "memory_on": s.memory is not None, "archive_on": s.archive is not None}
            ids, tok, sections, i = lp["ids"], self.ai.tok, [], 0
            while i < len(ids):
                kind = BLOCKS.get(ids[i])
                if kind is None:                                    # cannot happen with a prompt built by build_prompt(); shown as it is
                    sections.append({"kind": "other", "label": "Other", "tokens": 1, "text": ""})
                    i += 1
                    continue
                j = i + 1
                while j < len(ids) and ids[j] != END and ids[j] not in BLOCKS:
                    j += 1
                closed = j < len(ids) and ids[j] == END
                body = ids[i + 1:j]
                try:
                    text = tok.decode(body)
                except Exception:
                    text = ""
                sections.append({"kind": kind[0], "label": kind[1], "tokens": j + (1 if closed else 0) - i, "text": self._cut(text), "open": not closed})
                i = j + (1 if closed else 0)
            msgs = sum(1 for x in sections if x["kind"] in ("user", "assistant") and not x.get("open"))
            recalled = []
            mem = s.memory
            for mid in lp["memory"]:
                m = mem.memories.get(mid) if mem is not None else None
                if m is not None:
                    recalled.append({"id": mid, "text": self._cut(m["text"], 400), "status": m.get("status"), "emotion": m.get("emotion"), "pinned": bool(m.get("pinned"))})
            old = []
            if s.archive is not None:
                byid = {it["id"]: it for it in s.archive.items}
                for aid in lp["archive"]:
                    it = byid.get(aid)
                    if it is not None:
                        old.append({"id": aid, "user": self._cut(it["user"], 200), "ai": self._cut(it["ai"], 200), "when": time.strftime("%Y-%m-%d", time.localtime(it["t"]))})
            used = sum(x["tokens"] for x in sections)
            return {"has_prompt": True, "memory_on": mem is not None, "archive_on": s.archive is not None, "sections": sections, "tokens": used,
                    "budget": lp["budget"], "window": s.ctx, "reply_max": s.settings["max_new"], "messages_in": msgs,
                    "messages_left_out": max(lp["history"] - msgs, 0), "memories_unfit": lp["unfit"], "memory_budget": s.memory_tokens, "memories": recalled, "old_chats": old, "age_seconds": int(time.time() - lp["when"]),
                    "running": self.job is not None and self.job.state == "running"}

    # ---------------------------------------------------------------- U1b: settings, clear, save
    def _idle(self):
        if self.ai is None:
            raise ChatProblem("Load a model first.", 409)
        if self.job is not None and self.job.state == "running":
            raise ChatProblem("The AI is still writing a reply. Press Stop first.", 409)

    def settings(self, body):
        """Same limits as the typed /temp and /max. Only these three numbers can change; nothing else is reachable from here."""
        with self.lock:
            self._idle()
            s = self.ai.session
            new = {}
            for key, lo, hi, kind in (("temperature", 0, 2, float), ("top_k", 0, 1000, int), ("max_new", 1, s.ctx - 8, int)):
                if key not in body or body[key] is None:
                    continue
                v = body[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v <= hi) or (kind is int and v != int(v)):
                    raise ChatProblem("%s must be %s from %s to %s." % (key, "a whole number" if kind is int else "a number", lo, hi))
                new[key] = kind(v)
            if not new:
                raise ChatProblem("Nothing to change.")
            s.settings.update(new)
            return self.state()

    def reset(self):
        """Clears the conversation on this page and in the AI (not memory.json, not mood.json). Uses AI.reset(), the owner-side method."""
        with self.lock:
            self._idle()
            self.ai.reset()
            self.job = self.last_prompt = None
            return self.state()

    def save(self):
        """Writes the conversation to saved_chats/chat_DATE_TIME.txt (a new file; nothing is overwritten). The name is made here, never taken from the page."""
        with self.lock:
            self._idle()
            s = self.ai.session
            if not s.history:
                raise ChatProblem("Nothing to save yet.")
            try:
                folder = self.guard.resolve("saved_chats")
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                raise ChatProblem(self._say(e))
            stem = time.strftime("chat_%Y%m%d_%H%M%S", time.localtime())
            name, n = stem + ".txt", 1
            while os.path.exists(os.path.join(folder, name)) and n < 100:      # two saves in one second: the second gets _2, _3 ...
                n += 1
                name = "%s_%d.txt" % (stem, n)
            old, s.save_dir = s.save_dir, folder
            try:
                msg = s.save(name)                                  # the chat screen's own save: refuses to overwrite
            finally:
                s.save_dir = old
            if not msg.startswith("Saved"):
                raise ChatProblem(self._say(msg).replace(folder, "saved_chats"))
            return {"saved": "saved_chats/" + name}

    def stop(self):
        with self.lock:
            if self.job is not None and self.job.state == "running":
                self.job.stop.set()
                return {"stopping": True}
            return {"stopping": False}
