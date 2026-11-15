"""The Memory page of the web UI (stage U2a): list, search, add, edit, pin, forget, and the maintenance plan with a preview.

Owner-side only. These routes are for the person at the keyboard. The chat routes (ui_chat.py) never touch memory: a chat
message goes through say(), so "/remember ..." typed in the chat is only text and writes nothing.

  * One file: memory.json in the project folder (through PathGuard). It is read fresh for every request, so a change made
    by the terminal or the chat program is seen at once, and a save that would overwrite a newer file is refused (MemoryStore does that).
  * Every change is logged (action + ids only, never the memory text) and saved the same way the command line does it.
  * `forget` needs the memory id sent back as `confirm`, and removes the old text from the backup file too (as `/forget` does).
  * Maintenance is two steps: GET /api/memory/plan shows what WOULD happen and gives a plan_id; POST /api/memory/apply only
    carries out a plan whose id still matches a fresh plan. If memory changed in between, nothing is applied.
"""
import hashlib
import json
import re
import threading

from . import memory as MEM

MAX_LIST = 500
STATUS_CHOICES = ("all",) + MEM.STATUSES
SORTS = ("recent", "oldest", "strength", "importance")
_ID = re.compile(r"^m\d{1,9}$")


class MemProblem(Exception):
    """A plain sentence for the page. code is the HTTP status to answer with."""

    def __init__(self, msg, code=400):
        Exception.__init__(self, msg)
        self.code = code


def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise MemProblem("%s must be a number from 0 to 1." % name)
    if not 0 <= float(v) <= 1:
        raise MemProblem("%s must be a number from 0 to 1 (0.5 is normal, 0.9 is very important)." % name)
    return float(v)


class MemoryHub:
    def __init__(self, guard):
        self.guard = guard
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _path(self):
        return self.guard.resolve("memory.json")

    def _store(self):
        try:
            return MEM.MemoryStore(self._path())
        except MEM.MemoryProblem as e:
            raise MemProblem(str(e), 409)

    def _out(self, store, mem):
        now = store.clock()
        row = {"id": mem["id"], "text": mem["text"], "status": mem["status"], "emotion": mem["emotion"], "importance": mem["importance"],
               "strength": round(store.strength(mem, now), 3), "pinned": bool(mem.get("pinned")), "uses": mem.get("uses", 0),
               "timestamp": mem["timestamp"], "last_used": mem["last_used"], "source": mem.get("source", ""),
               "nodes": [{"id": n, "name": store.nodes[n]["name"], "kind": store.nodes[n]["kind"]} for n in mem["nodes"] if n in store.nodes]}
        for k in ("summary_of", "merged_into", "archived_into"):
            if mem.get(k):
                row[k] = mem[k]
        return row

    @staticmethod
    def _mid(body):
        mid = body.get("id")
        if not isinstance(mid, str) or not _ID.match(mid.strip().lower()):
            raise MemProblem("Give the memory id, like m0003.")
        return mid.strip().lower()

    def _run(self, fn):
        try:
            with self.lock:
                return fn()
        except MEM.MemoryConflict as e:
            raise MemProblem(str(e), 409)
        except MEM.MemoryProblem as e:
            raise MemProblem(str(e), 400)

    # ------------------------------------------------------------------ read
    def list(self, q):
        text = q.get("q", "")
        status = q.get("status", "all")
        sort = q.get("sort", "recent")
        if len(text) > 200:
            raise MemProblem("The search text is too long (200 characters at most).")
        if status not in STATUS_CHOICES:
            raise MemProblem("status must be one of: %s." % ", ".join(STATUS_CHOICES))
        if sort not in SORTS:
            raise MemProblem("sort must be one of: %s." % ", ".join(SORTS))
        try:
            limit = int(q.get("limit", 100))
            offset = int(q.get("offset", 0))
        except ValueError:
            raise MemProblem("limit and offset must be whole numbers.")
        if not 1 <= limit <= MAX_LIST or offset < 0:
            raise MemProblem("limit must be 1 to %d and offset 0 or more." % MAX_LIST)

        def go():
            path = self._path()
            import os
            exists = os.path.exists(path)
            store = self._store()
            now = store.clock()
            mems = list(store.memories.values())
            counts = {s: sum(1 for m in mems if m["status"] == s) for s in MEM.STATUSES}
            words = text.casefold().split()
            if status != "all":
                mems = [m for m in mems if m["status"] == status]
            if words:
                def hay(m):
                    return " ".join([m["id"], m["text"], m["emotion"]] + [store.nodes[n]["name"] for n in m["nodes"] if n in store.nodes]).casefold()
                mems = [m for m in mems if all(w in hay(m) for w in words)]
            key = {"recent": lambda m: (m["timestamp"], m["id"]), "oldest": lambda m: (m["timestamp"], m["id"]),
                   "strength": lambda m: (store.strength(m, now), m["id"]), "importance": lambda m: (m["importance"], m["id"])}[sort]
            mems.sort(key=key, reverse=(sort != "oldest"))
            shown = mems[offset:offset + limit]
            return {"file": "memory.json", "exists": exists, "total": len(store.memories), "counts": counts, "matches": len(mems), "offset": offset,
                    "memories": [self._out(store, m) for m in shown], "overview": store.overview(now), "emotions": sorted({m["emotion"] for m in store.memories.values()}),
                    "kinds": list(MEM.KINDS), "max_text": MEM.MAX_TEXT}
        return self._run(go)

    def _plan(self, store):
        plan = store.plan_maintenance()
        pid = hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return plan, pid

    def plan(self):
        """What maintenance WOULD do. Changes nothing."""
        def go():
            store = self._store()
            plan, pid = self._plan(store)

            def brief(mid):
                m = store.memories[mid]
                return {"id": mid, "text": m["text"][:240], "strength": round(store.strength(m), 3), "importance": m["importance"]}
            out = {"plan_id": pid, "fade": [brief(i) for i in plan["fade"]], "revive": [brief(i) for i in plan["revive"]],
                   "duplicates": [{"keep": brief(k), "drop": brief(d), "similarity": s} for k, d, s in plan["duplicates"]],
                   "summaries": [{"node": store.nodes[s["node"]]["name"], "count": len(s["ids"]), "ids": s["ids"], "text": s["text"][:400]} for s in plan["summaries"]]}
            out["empty"] = not (plan["fade"] or plan["revive"] or plan["duplicates"] or plan["summaries"])
            return out
        return self._run(go)

    # ------------------------------------------------------------------ write (owner only)
    def add(self, body):
        text = body.get("text")
        if not isinstance(text, str):
            raise MemProblem("Write what to remember.")
        emotion = body.get("emotion", "neutral")
        if not isinstance(emotion, str):
            raise MemProblem("The emotion tag must be one word, e.g. joy.")
        importance = _num(body.get("importance", 0.5), "Importance")
        pinned = body.get("pinned", False)
        if not isinstance(pinned, bool):
            raise MemProblem("pinned must be true or false.")

        def go():
            store = self._store()
            mem = store.add(text, emotion=emotion, importance=importance, source="owner", pinned=pinned)
            store.log("add", [mem["id"]])
            store.save()
            return {"memory": self._out(store, mem)}
        return self._run(go)

    def edit(self, body):
        mid = self._mid(body)
        has = [k for k in ("text", "emotion", "importance", "pinned") if k in body]
        if not has:
            raise MemProblem("Nothing to change: send text, emotion, importance or pinned.")
        if "text" in body and not isinstance(body["text"], str):
            raise MemProblem("The text must be a string.")
        if "emotion" in body and not isinstance(body["emotion"], str):
            raise MemProblem("The emotion tag must be one word, e.g. joy.")
        if "pinned" in body and not isinstance(body["pinned"], bool):
            raise MemProblem("pinned must be true or false.")
        imp = _num(body["importance"], "Importance") if "importance" in body else None

        def go():
            store = self._store()
            mem = store.get(mid)
            store.edit(mid, text=body.get("text"), emotion=body.get("emotion"))
            if imp is not None or "pinned" in body:
                store.set_importance(mid, mem["importance"] if imp is None else imp, body.get("pinned"))
            store.log("edit" if ("text" in body or "emotion" in body) else "set", [mid])
            store.save()
            return {"memory": self._out(store, mem)}
        return self._run(go)

    def forget(self, body):
        mid = self._mid(body)
        if body.get("confirm") != mid:
            raise MemProblem("To forget a memory for good, send its id again as confirm (the page asks you to press the button twice).")

        def go():
            store = self._store()
            store.forget(mid)
            store.log("forget", [mid])
            store.save(keep_backup=False)            # the old text must not survive in memory.prev.json
            return {"forgotten": mid, "total": len(store.memories)}
        return self._run(go)

    def apply(self, body):
        pid = body.get("plan_id")
        if not isinstance(pid, str) or not pid:
            raise MemProblem("Preview the plan first, then apply it with its plan_id.")

        def go():
            store = self._store()
            plan, now_id = self._plan(store)
            if pid != now_id:
                raise MemProblem("Memory changed since that preview. Nothing was applied. Preview again.", 409)
            if not (plan["fade"] or plan["revive"] or plan["duplicates"] or plan["summaries"]):
                raise MemProblem("There is nothing to do.")
            done = store.apply_maintenance(plan)
            store.log("maintain", [], note=json.dumps(done))
            store.save()
            return {"done": done}
        return self._run(go)
