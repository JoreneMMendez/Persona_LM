"""Stage U2a: the memory routes of the web UI (list, search, add, edit, pin, forget, maintenance preview + apply).

Rules (each has a test in tests/test_ui_u2a.py):
  * Only the file memory.json in the project folder, opened through PathGuard. Nothing else is read or written.
  * These routes are for the OWNER at the keyboard (token + Host + Origin checks, like every /api route). The chat route still uses
    say(), so text the model writes can never reach them. No model text is ever passed to any function in this file.
  * Every change goes through MemoryStore (same validation, atomic save, conflict check as the typed commands) and is logged
    without the text (memory.log.jsonl), like the CLI does.
  * forget needs the id typed back as `confirm` (the page sends it after its two-step button) and, like the CLI, saves with
    keep_backup=False so the old text does not survive in memory.prev.json.
  * Maintenance is a PLAN first. Applying needs the plan_id of the preview; if memory changed since, nothing is applied.
"""
import hashlib
import json
import re
import threading
import time

from . import memory as MEM

DEFAULT_LIMIT = 100
MAX_LIMIT = 500
MAX_QUERY = 200
STATUS_FILTERS = ("all",) + MEM.STATUSES
SORTS = ("newest", "oldest", "strength", "importance")
_ID = re.compile(r"^m\d{1,8}$")


class MemoryHubProblem(Exception):
    """A plain sentence for the page. code is the HTTP status to answer with."""

    def __init__(self, msg, code=400):
        Exception.__init__(self, msg)
        self.code = code


def _number(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise MemoryHubProblem("%s must be a number from 0 to 1." % what)
    return float(v)


class MemoryHub:
    def __init__(self, guard):
        self.guard = guard
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- helpers
    def _clean(self, e):
        return str(e).replace(self.guard.root, ".")                 # no absolute folder names in answers

    def _path(self):
        try:
            return self.guard.resolve("memory.json")
        except Exception as e:
            raise MemoryHubProblem(self._clean(e), 403)

    def _open(self):
        """(store, file_exists). A missing file is an empty memory; it is created by the first add."""
        import os
        path = self._path()
        try:
            return MEM.MemoryStore(path), os.path.exists(path)
        except MEM.MemoryProblem as e:
            raise MemoryHubProblem(self._clean(e), 400)

    def _save(self, store, action, ids, forget=False, note=None):
        try:
            store.save(keep_backup=not forget)
            store.log(action, ids, note=note)
        except MEM.MemoryConflict as e:
            raise MemoryHubProblem(self._clean(e), 409)
        except MEM.MemoryProblem as e:
            raise MemoryHubProblem(self._clean(e), 400)
        except OSError as e:
            raise MemoryHubProblem("Could not write the memory file: " + self._clean(e), 500)

    @staticmethod
    def _mid(v):
        if not isinstance(v, str) or not _ID.match(v.strip().lower()):
            raise MemoryHubProblem("Give a memory id like m0003.")
        return v.strip().lower()

    @staticmethod
    def _item(store, mem, now):
        return {"id": mem["id"], "text": mem["text"], "emotion": mem["emotion"], "importance": mem["importance"],
                "strength": round(store.strength(mem, now), 3), "pinned": bool(mem.get("pinned")), "status": mem["status"],
                "uses": mem.get("uses", 0), "timestamp": mem["timestamp"], "last_used": mem.get("last_used"), "source": mem.get("source", ""),
                "nodes": [{"name": store.nodes[n]["name"], "kind": store.nodes[n]["kind"]} for n in mem["nodes"] if n in store.nodes],
                "summary_of": list(mem.get("summary_of") or []), "merged_into": mem.get("merged_into")}

    # ---------------------------------------------------------------- list
    def list(self, q="", status="all", sort="newest", pinned=False, limit=DEFAULT_LIMIT, offset=0):
        if not isinstance(q, str) or len(q) > MAX_QUERY:
            raise MemoryHubProblem("The search text is too long (limit %d characters)." % MAX_QUERY)
        if status not in STATUS_FILTERS:
            raise MemoryHubProblem("status must be one of: " + ", ".join(STATUS_FILTERS))
        if sort not in SORTS:
            raise MemoryHubProblem("sort must be one of: " + ", ".join(SORTS))
        if not (isinstance(limit, int) and 1 <= limit <= MAX_LIMIT and isinstance(offset, int) and offset >= 0):
            raise MemoryHubProblem("limit must be 1 to %d and offset 0 or more." % MAX_LIMIT)
        with self.lock:
            store, exists = self._open()
            now = time.time()
            mems = list(store.memories.values())
            counts = {s: sum(1 for m in mems if m["status"] == s) for s in MEM.STATUSES}
            words = [w.casefold() for w in q.split() if w.strip()]
            out = []
            for m in mems:
                if status != "all" and m["status"] != status:
                    continue
                if pinned and not m.get("pinned"):
                    continue
                if words:
                    hay = (m["text"] + " " + m["id"] + " " + m["emotion"] + " " + " ".join(store.nodes[n]["name"] for n in m["nodes"] if n in store.nodes)).casefold()
                    if not all(w in hay for w in words):
                        continue
                out.append(m)
            keys = {"newest": lambda m: (m["timestamp"], m["id"]), "oldest": lambda m: (m["timestamp"], m["id"]),
                    "strength": lambda m: (store.strength(m, now), m["id"]), "importance": lambda m: (m["importance"], m["id"])}
            out.sort(key=keys[sort], reverse=sort != "oldest")
            page = out[offset:offset + limit]
            emotions = sorted({m["emotion"] for m in mems})
            return {"exists": exists, "items": [self._item(store, m, now) for m in page], "matched": len(out), "total": len(mems),
                    "counts": counts, "nodes": len(store.nodes), "emotions": emotions, "pinned": sum(1 for m in mems if m.get("pinned")),
                    "offset": offset, "limit": limit}

    # ---------------------------------------------------------------- add / edit / forget
    def add(self, body):
        text = body.get("text")
        if not isinstance(text, str):
            raise MemoryHubProblem("Write what to remember.")
        emotion = body.get("emotion", "neutral")
        if not isinstance(emotion, str):
            raise MemoryHubProblem("The emotion tag must be one lowercase word.")
        importance = _number(body.get("importance", 0.5), "Importance")
        pinned = body.get("pinned", False)
        if not isinstance(pinned, bool):
            raise MemoryHubProblem("pinned must be true or false.")
        with self.lock:
            store, _ = self._open()
            try:
                mem = store.add(text, emotion=emotion or "neutral", importance=importance, source="owner", pinned=pinned)
            except MEM.MemoryProblem as e:
                raise MemoryHubProblem(self._clean(e), 400)
            self._save(store, "add", [mem["id"]])
            return {"memory": self._item(store, mem, time.time())}

    def edit(self, body):
        mid = self._mid(body.get("id"))
        fields = {k: body[k] for k in ("text", "emotion", "importance", "pinned") if k in body}
        if not fields:
            raise MemoryHubProblem("Nothing to change: send text, emotion, importance or pinned.")
        if "pinned" in fields and not isinstance(fields["pinned"], bool):
            raise MemoryHubProblem("pinned must be true or false.")
        if "emotion" in fields and not isinstance(fields["emotion"], str):
            raise MemoryHubProblem("The emotion tag must be one lowercase word.")
        if "importance" in fields:
            _number(fields["importance"], "Importance")
        if "text" in fields and not isinstance(fields["text"], str):
            raise MemoryHubProblem("The text must be a string.")
        with self.lock:
            store, exists = self._open()
            if not exists:
                raise MemoryHubProblem("There is no memory file yet.", 404)
            try:
                mem = store.get(mid)
            except MEM.MemoryProblem:
                raise MemoryHubProblem("There is no memory %s." % mid, 404)
            nodes_backup = json.loads(json.dumps(store.nodes))
            before = json.loads(json.dumps(mem))
            try:
                if "text" in fields:
                    text = " ".join(fields["text"].split())
                    if not text:
                        raise MEM.MemoryProblem("There is nothing to remember: the text is empty.")
                    if len(text) > MEM.MAX_TEXT:
                        raise MEM.MemoryProblem("That memory is %d characters; the limit is %d." % (len(text), MEM.MAX_TEXT))
                    _, old_auto = store._resolve_links(mem["text"])               # links the OLD text produced by itself ...
                    extra = [n for n in mem["nodes"] if n not in old_auto]       # ... so links added by hand (CLI --node) are kept
                    clean, ids = store._resolve_links(text)
                    mem["text"] = clean
                    mem["nodes"] = ids + [n for n in extra if n not in ids]
                if "emotion" in fields:
                    emo = fields["emotion"].strip().lower() or "neutral"
                    if not MEM._EMOTION.match(emo):
                        raise MEM.MemoryProblem("The emotion tag must be one lowercase word (letters, digits, '-'), up to 24 characters, e.g. joy, sadness, stress.")
                    mem["emotion"] = emo
                if "importance" in fields or "pinned" in fields:
                    store.set_importance(mid, fields.get("importance", mem["importance"]), fields.get("pinned"))
                store.prune_nodes()
            except MEM.MemoryProblem as e:
                store.data["nodes"] = nodes_backup                                # a rejected edit leaves nothing half done
                mem.clear(); mem.update(before)
                raise MemoryHubProblem(self._clean(e), 400)
            self._save(store, "set", [mid])
            return {"memory": self._item(store, mem, time.time())}

    def forget(self, body):
        mid = self._mid(body.get("id"))
        if body.get("confirm") != mid:
            raise MemoryHubProblem("To delete a memory for good, send its id again as confirm (the page does this after the second click).")
        with self.lock:
            store, exists = self._open()
            if not exists:
                raise MemoryHubProblem("There is no memory file yet.", 404)
            try:
                store.get(mid)
            except MEM.MemoryProblem:
                raise MemoryHubProblem("There is no memory %s." % mid, 404)
            store.forget(mid)
            self._save(store, "forget", [mid], forget=True)
            return {"forgotten": mid}

    # ---------------------------------------------------------------- maintenance
    @staticmethod
    def _plan_id(plan):
        raw = json.dumps({"fade": plan["fade"], "revive": plan["revive"], "duplicates": [[a, b] for a, b, _ in plan["duplicates"]],
                          "summaries": [[s["node"], s["ids"]] for s in plan["summaries"]]}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def plan(self):
        with self.lock:
            store, exists = self._open()
            if not exists:
                raise MemoryHubProblem("There is no memory file yet, so there is nothing to tidy.", 404)
            plan = store.plan_maintenance()
            text = lambda mid: store.memories[mid]["text"]
            return {"plan_id": self._plan_id(plan),
                    "fade": [{"id": i, "text": text(i)} for i in plan["fade"]],
                    "revive": [{"id": i, "text": text(i)} for i in plan["revive"]],
                    "duplicates": [{"keep": a, "drop": b, "similarity": s, "keep_text": text(a), "drop_text": text(b)} for a, b, s in plan["duplicates"]],
                    "summaries": [{"node": store.nodes[s["node"]]["name"], "ids": s["ids"], "text": s["text"]} for s in plan["summaries"]],
                    "empty": not (plan["fade"] or plan["revive"] or plan["duplicates"] or plan["summaries"])}

    def maintain(self, body):
        pid = body.get("plan_id")
        if not isinstance(pid, str) or not pid:
            raise MemoryHubProblem("Preview the plan first, then apply it with its plan_id.")
        with self.lock:
            store, exists = self._open()
            if not exists:
                raise MemoryHubProblem("There is no memory file yet.", 404)
            plan = store.plan_maintenance()
            if not (plan["fade"] or plan["revive"] or plan["duplicates"] or plan["summaries"]):
                raise MemoryHubProblem("Nothing to tidy right now.")
            if pid != self._plan_id(plan):
                raise MemoryHubProblem("Memory changed since the preview. Nothing was applied: preview again.", 409)
            try:
                done = store.apply_maintenance(plan)
            except MEM.MemoryProblem as e:
                raise MemoryHubProblem(self._clean(e), 400)
            self._save(store, "maintain", [], note=json.dumps(done))
            return {"done": done}
