"""Stage U2b: graph, mood and Person card routes (persona_lm/ui_mind.py). Real loopback sockets, no browser."""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import emotion as EMO
from persona_lm import memory as MEM
from persona_lm import person as PER
from test_ui_u0 import SECRET, Base


class MindBase(Base):
    def setUp(self):
        for n in ("memory.json", "mood.json", "person_card.md", "person_card.prev.md", "memory.prev.json"):
            p = os.path.join(self.root, n)
            if os.path.exists(p):
                os.remove(p)

    def p(self, n):
        return os.path.join(self.root, n)

    def get(self, path, **kw):
        st, _, d = self.api("GET", path, **kw)
        return st, json.loads(d)

    def post(self, path, body=None, **kw):
        st, _, d = self.api("POST", path, {} if body is None else body, **kw)
        return st, json.loads(d)

    def card(self):
        c = PER.new_card("Jorene")
        c.add_owner("Habits", "studies at night")
        c.section("Relationships")[1].insert(0, {"kind": "fact", "tag": "auto", "text": "Maria: my sister"})
        PER.save(c, self.p("person_card.md"))

    def text(self):
        with open(self.p("person_card.md"), encoding="utf-8") as f:
            return f.read()


class Graph(MindBase):
    def test_missing_file(self):
        st, d = self.get("/api/graph")
        self.assertEqual((st, d["exists"], d["nodes"]), (200, False, []))

    def test_nodes_edges_and_scores(self):
        s = MEM.MemoryStore(self.p("memory.json"))
        s.add("[[person:Maria]] loves [[topic:Mango]]", importance=0.9)
        s.add("[[person:Maria]] lives in [[topic:Cebu]]", importance=0.5)
        s.add("old [[topic:Rain]] note", importance=0.5)["status"] = "dormant"
        s.save()
        st, d = self.get("/api/graph")
        self.assertEqual(st, 200, d)
        names = {n["name"]: n for n in d["nodes"]}
        self.assertEqual(d["nodes"][0]["name"], "Maria")                      # biggest first
        self.assertEqual(names["Maria"]["memories"], 2); self.assertEqual(names["Maria"]["kind"], "person")
        self.assertEqual(names["Rain"]["memories"], 0)                         # dormant memories do not count
        ids = {n["name"]: n["id"] for n in d["nodes"]}
        pairs = {(e["a"], e["b"]): e for e in d["edges"]}
        self.assertEqual(len(d["edges"]), 2)
        self.assertTrue(all(e["a"] < e["b"] and e["count"] == 1 for e in d["edges"]))
        self.assertIn(tuple(sorted((ids["Maria"], ids["Mango"]))), pairs)

    def test_big_graph_is_cut_to_the_biggest_nodes(self):
        s = MEM.MemoryStore(self.p("memory.json"))
        for i in range(160):
            s.add("[[topic:Topic%03d]] note number" % i, importance=0.5 + i / 1000.0)
        s.save()
        d = self.get("/api/graph")[1]
        self.assertEqual((d["shown"], d["total_nodes"], len(d["nodes"])), (150, 160, 150))
        self.assertEqual(d["nodes"][0]["name"], "Topic159")

    def test_broken_file(self):
        with open(self.p("memory.json"), "w") as f:
            f.write("{bad")
        st, d = self.get("/api/graph")
        self.assertEqual(st, 400); self.assertNotIn(self.root, d["error"])


class Mood(MindBase):
    def test_missing(self):
        d = self.get("/api/mood")[1]
        self.assertFalse(d["exists"]); self.assertIn("joy", d["labels"])

    def test_values_history_and_file_untouched(self):
        m = EMO.Mood(self.p("mood.json"))
        m.update(EMO.Tag("joy", 0.9, {}, []), now=time.time() - 60)
        m.update(EMO.Tag("sadness", 0.8, {}, []), now=time.time() - 30)
        m.save()
        with open(self.p("mood.json"), "rb") as f:
            before = f.read()
        d = self.get("/api/mood")[1]
        self.assertTrue(d["exists"]); self.assertEqual([h["label"] for h in d["history"]], ["joy", "sadness"])
        self.assertEqual(d["turns"], 2); self.assertIn(d["label"], d["labels"])
        self.assertTrue(-1 <= d["valence"] <= 1 and -1 <= d["arousal"] <= 1)
        with open(self.p("mood.json"), "rb") as f:
            self.assertEqual(f.read(), before)                                  # reading never writes the mood

    def test_broken_file(self):
        with open(self.p("mood.json"), "w") as f:
            f.write("nope")
        st, d = self.get("/api/mood")
        self.assertEqual(st, 400); self.assertNotIn(self.root, d["error"])


class Card(MindBase):
    def test_no_card_is_an_empty_view_and_add_creates_it(self):
        d = self.get("/api/person")[1]
        self.assertFalse(d["exists"])
        st, d = self.post("/api/person/add", {"section": "values", "text": "  honesty   first "})
        self.assertEqual(st, 200, d)
        vals = [s for s in d["sections"] if s["title"] == "Values"][0]
        self.assertEqual(vals["lines"], [{"tag": "owner", "text": "honesty first"}])
        self.assertIn("- honesty first", self.text())

    def test_view_sections_and_tags(self):
        self.card()
        d = self.get("/api/person")[1]
        self.assertTrue(d["exists"])
        rel = [s for s in d["sections"] if s["title"] == "Relationships"][0]
        self.assertTrue(rel["auto"]); self.assertEqual(rel["lines"][0], {"tag": "auto", "text": "Maria: my sister"})
        self.assertNotIn("<!--", json.dumps(d))                                # comments are not shown as lines

    def test_edit_text_makes_it_the_owners_line_and_keeps_a_backup(self):
        self.card()
        st, d = self.post("/api/person/edit", {"section": "Relationships", "index": 0, "old": "Maria: my sister", "text": "Maria: my older sister"})
        self.assertEqual(st, 200, d)
        self.assertIn("- Maria: my older sister", self.text()); self.assertNotIn("[auto] Maria", self.text())
        self.assertTrue(os.path.exists(self.p("person_card.prev.md")))
        st, d = self.post("/api/person/edit", {"section": "Relationships", "index": 0, "old": "Maria: my older sister", "tag": "no"})
        self.assertEqual(st, 200); self.assertIn("- [no] Maria: my older sister", self.text())

    def test_stale_or_wrong_lines_change_nothing(self):
        self.card()
        before = self.text()
        bad = ({"section": "Habits", "index": 0, "old": "something else", "text": "x"}, {"section": "Habits", "index": 5, "old": "studies at night", "text": "x"},
               {"section": "Habits", "index": -1, "old": "studies at night", "text": "x"}, {"section": "Habits", "index": True, "old": "studies at night", "text": "x"},
               {"section": "Nope", "index": 0, "old": "a", "text": "x"}, {"section": "Habits", "index": 0, "old": "studies at night"},
               {"section": "Habits", "index": 0, "old": "studies at night", "text": ""}, {"section": "Habits", "index": 0, "old": "studies at night", "tag": "evil"},
               {"section": "Habits", "index": 0, "old": "studies at night", "text": "y" * 401}, {"section": "Habits", "index": 0, "text": "x"})
        for b in bad:
            st, d = self.post("/api/person/edit", b)
            self.assertIn(st, (400, 409), b)
        self.assertEqual(self.text(), before)
        self.assertEqual(self.post("/api/person/edit", {"section": "Habits", "index": 0, "old": "studies at night", "text": "x"})[0], 200)

    def test_remove_one_line(self):
        self.card()
        st, d = self.post("/api/person/remove", {"section": "Habits", "index": 0, "old": "studies at night"})
        self.assertEqual(st, 200, d)
        self.assertNotIn("studies at night", self.text()); self.assertIn("Maria: my sister", self.text())
        self.assertEqual(self.post("/api/person/remove", {"section": "Habits", "index": 0, "old": "studies at night"})[0], 409)

    def test_edit_without_a_card(self):
        self.assertEqual(self.post("/api/person/edit", {"section": "Habits", "index": 0, "old": "a", "text": "b"})[0], 404)
        self.assertFalse(os.path.exists(self.p("person_card.md")))


class Safety(MindBase):
    def test_token_host_origin_and_methods(self):
        self.card()
        for path in ("/api/graph", "/api/mood", "/api/person"):
            self.assertEqual(self.api("GET", path, token=False)[0], 401)
            self.assertEqual(self.api("GET", path, host="evil.example:1")[0], 403)
            self.assertEqual(self.api("POST", path, {})[0], 404)               # reading is GET only
        before = self.text()
        for path in ("/api/person/add", "/api/person/edit", "/api/person/remove"):
            self.assertEqual(self.api("POST", path, {"section": "Habits", "text": "x"}, token="wrong")[0], 401)
            self.assertEqual(self.api("POST", path, {"section": "Habits", "text": "x"}, headers={"Origin": "http://evil.example"})[0], 403)
            self.assertEqual(self.api("GET", path)[0], 404)
            self.assertEqual(self.api("DELETE", path, {})[0], 405)
        self.assertEqual(self.text(), before)

    def test_no_extra_routes_no_mood_writes_no_private_reads(self):
        for p in ("/api/mood/reset", "/api/mood/set", "/api/person/build", "/api/person/raw", "/api/graph/export", "/api/person?path=data_msgr/private/raw_chat_notes.txt"):
            for m in ("GET", "POST"):
                st, _, body = self.api(m, p, {} if m == "POST" else None)
                if "?" not in p:
                    self.assertEqual(st, 404, (m, p))
                self.assertNotIn(SECRET.encode(), body)

    def test_a_card_link_into_private_is_refused(self):
        if not self.links:
            self.skipTest("no symlinks")
        try:
            os.symlink(os.path.join(self.root, "data_msgr", "private", "raw_chat_notes.txt"), self.p("person_card.md"))
        except OSError:
            self.skipTest("no symlinks")
        try:
            st, _, body = self.api("GET", "/api/person")
            self.assertIn(st, (400, 403)); self.assertNotIn(SECRET.encode(), body)
        finally:
            os.remove(self.p("person_card.md"))


if __name__ == "__main__":
    unittest.main()
