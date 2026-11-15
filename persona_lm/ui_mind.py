"""Stage U2b: graph, mood and Person card routes of the web UI.

  GET  /api/graph          nodes and links of memory.json (read only; active memories only)
  GET  /api/mood           the mood in mood.json now (drift applied in RAM only) and its history (read only)
  GET  /api/person         person_card.md as sections of lines
  POST /api/person/add|edit|remove   change ONE line of the card, by the owner at the keyboard

Rules (tests in tests/test_ui_u2b.py): only memory.json, mood.json and person_card.md, each through PathGuard; the mood file is never
written; a card line is changed only when the `old` text the page saw still matches (else 409); editing an [auto] line makes it the
owner's line (tag owner) so a rebuild does not replace it; a card edit keeps the previous file as person_card.prev.md like the CLI.
Nothing the model writes can reach these routes.
"""
import os
import threading
import time

from . import emotion as EMO
from . import memory as MEM
from . import person as PER

MAX_NODES = 150
MAX_LINE = 400


class MindProblem(Exception):
    def __init__(self, msg, code=400):
        Exception.__init__(self, msg)
        self.code = code


class MindHub:
    def __init__(self, guard):
        self.guard = guard
        self.lock = threading.Lock()

    def _clean(self, e):
        return str(e).replace(self.guard.root, ".")

    def _path(self, name):
        try:
            return self.guard.resolve(name)
        except Exception as e:
            raise MindProblem(self._clean(e), 403)

    # ---------------------------------------------------------------- graph
    def graph(self):
        path = self._path("memory.json")
        if not os.path.exists(path):
            return {"exists": False, "nodes": [], "edges": [], "total_nodes": 0, "shown": 0}
        try:
            store = MEM.MemoryStore(path)
        except MEM.MemoryProblem as e:
            raise MindProblem(self._clean(e))
        now = time.time()
        scores = store.node_scores(now)
        counts = {}
        for m in store.active():
            for n in m["nodes"]:
                counts[n] = counts.get(n, 0) + 1
        top = sorted(store.nodes.values(), key=lambda n: (-scores[n["id"]], n["id"]))[:MAX_NODES]
        keep = {n["id"] for n in top}
        edges = [{"a": a, "b": b, "count": e["count"], "weight": round(e["weight"], 3)}
                 for (a, b), e in sorted(store.edges(now).items()) if a in keep and b in keep]
        return {"exists": True, "total_nodes": len(store.nodes), "shown": len(top), "edges": edges,
                "nodes": [{"id": n["id"], "name": n["name"], "kind": n["kind"], "score": scores[n["id"]], "memories": counts.get(n["id"], 0)} for n in top]}

    # ---------------------------------------------------------------- mood
    def mood(self):
        path = self._path("mood.json")
        labels = {k: list(v) for k, v in EMO.LABELS.items()}
        if not os.path.exists(path):
            return {"exists": False, "labels": labels}
        try:
            m = EMO.Mood(path)
        except EMO.EmotionProblem as e:
            raise MindProblem(self._clean(e))
        m._drift(time.time())                                   # in RAM only: the file is never written here
        return {"exists": True, "labels": labels, "label": m.label, "valence": round(m.valence, 3), "arousal": round(m.arousal, 3),
                "baseline": m.baseline, "intensity": round(m.intensity, 3), "turns": m.turns,
                "history": [{"label": str(h.get("label")), "confidence": h.get("confidence"), "time": h.get("time")} for h in m.history if isinstance(h, dict)]}

    # ---------------------------------------------------------------- person card
    def _card(self, must_exist):
        path = self._path("person_card.md")
        if os.path.exists(path):
            try:
                return PER.load(path), path, True
            except PER.PersonProblem as e:
                raise MindProblem(self._clean(e))
        if must_exist:
            raise MindProblem("There is no Person card yet.", 404)
        return PER.new_card(), path, False

    def person(self):
        card, _, exists = self._card(False)
        sections = []
        for title, items in card.sections:
            facts = [it for it in items if it["kind"] == "fact"]
            sections.append({"title": title, "auto": title in PER.AUTO_SECTIONS,
                             "lines": [{"tag": f["tag"], "text": f["text"]} for f in facts]})
        return {"exists": exists, "sections": sections, "tags": list(PER.TAGS)}

    def _save(self, card, path):
        try:
            PER.save(card, path)
        except OSError as e:
            raise MindProblem("Could not write the Person card: " + self._clean(e), 500)

    @staticmethod
    def _text(v):
        if not isinstance(v, str):
            raise MindProblem("Write the line as text.")
        t = " ".join(v.split())
        if not t:
            raise MindProblem("There is nothing to write: the text is empty.")
        if len(t) > MAX_LINE:
            raise MindProblem("A card line should be short (%d characters; at most %d)." % (len(t), MAX_LINE))
        return t

    @staticmethod
    def _section(v):
        if not isinstance(v, str):
            raise MindProblem("Give a section name.")
        try:
            return PER.Card._canonical(v)
        except PER.PersonProblem as e:
            raise MindProblem(str(e))

    def _fact(self, card, body):
        sec = self._section(body.get("section"))
        idx, old = body.get("index"), body.get("old")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or not isinstance(old, str):
            raise MindProblem("Say which line (index) and what it said (old).")
        facts = card.facts(sec)
        if idx >= len(facts) or facts[idx]["text"] != old:
            raise MindProblem("That line changed since you loaded the card. Nothing was changed: reload the page.", 409)
        return sec, facts[idx]

    def add(self, body):
        text = self._text(body.get("text"))
        sec = self._section(body.get("section"))
        with self.lock:
            card, path, _ = self._card(False)
            try:
                card.add_owner(sec, text)
            except PER.PersonProblem as e:
                raise MindProblem(str(e))
            self._save(card, path)
            return self.person()

    def edit(self, body):
        with self.lock:
            card, path, _ = self._card(True)
            sec, fact = self._fact(card, body)
            if "text" not in body and "tag" not in body:
                raise MindProblem("Nothing to change: send text or tag.")
            if "tag" in body and body["tag"] not in PER.TAGS:
                raise MindProblem("tag must be one of: " + ", ".join(PER.TAGS))
            if "text" in body:
                fact["text"] = self._text(body["text"])
                fact["tag"] = "owner"                           # your words are yours: a rebuild must not replace them
            if "tag" in body:
                fact["tag"] = body["tag"]
            self._save(card, path)
            return self.person()

    def remove(self, body):
        with self.lock:
            card, path, _ = self._card(True)
            sec, fact = self._fact(card, body)
            card.section(sec)[1].remove(fact)
            self._save(card, path)
            return self.person()
