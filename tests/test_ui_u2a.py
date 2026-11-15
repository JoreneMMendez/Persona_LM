"""Stage U2a: memory routes of the web UI (persona_lm/ui_memory.py). Real loopback sockets, no browser, no network."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import memory as MEM
from test_ui_u0 import SECRET, Base

DAY = 86400.0


class MemBase(Base):
    def setUp(self):
        self.mem_path = os.path.join(self.root, "memory.json")
        for suffix in ("", ".tmp"):
            if os.path.exists(self.mem_path + suffix):
                os.remove(self.mem_path + suffix)
        for extra in ("memory.prev.json", "memory.log.jsonl"):
            p = os.path.join(self.root, extra)
            if os.path.exists(p):
                os.remove(p)

    def seed(self, rows):
        """rows: (text, importance, age_days, pinned). Written like the CLI would."""
        import time
        t = time.time()
        s = MEM.MemoryStore(self.mem_path, clock=lambda: t)
        for text, imp, age, pinned in rows:
            m = s.add(text, importance=imp, when=t - age * DAY, pinned=pinned)
            m["last_used"] = m["timestamp"]
        s.save()
        return s

    def post(self, path, body=None, **kw):
        st, _, data = self.api("POST", path, {} if body is None else body, **kw)
        return st, json.loads(data)

    def get(self, path, **kw):
        st, _, data = self.api("GET", path, **kw)
        return st, json.loads(data)

    def disk(self):
        with open(self.mem_path, encoding="utf-8") as f:
            return json.load(f)


class ListAndSearch(MemBase):
    def test_missing_file_is_an_empty_list_not_an_error(self):
        st, d = self.get("/api/memory/list")
        self.assertEqual(st, 200, d)
        self.assertFalse(d["exists"]); self.assertEqual((d["items"], d["total"]), ([], 0))

    def test_list_fields_counts_and_nodes(self):
        self.seed([("[[person:Maria]] is my sister and loves mango", 0.8, 1, True), ("I live in Bacolod", 0.4, 2, False)])
        st, d = self.get("/api/memory/list")
        self.assertEqual(st, 200, d)
        self.assertTrue(d["exists"]); self.assertEqual((d["total"], d["matched"], d["pinned"]), (2, 2, 1))
        self.assertEqual(d["counts"]["active"], 2)
        top = d["items"][0]
        self.assertEqual(top["id"], "m0001"); self.assertTrue(top["pinned"])
        self.assertEqual(top["nodes"], [{"name": "Maria", "kind": "person"}])
        self.assertNotIn("[[", top["text"])
        for k in ("strength", "importance", "emotion", "status", "uses", "timestamp", "source"):
            self.assertIn(k, top)

    def test_search_words_ids_nodes_and_filters(self):
        self.seed([("[[person:Maria]] loves mango", 0.9, 3, False), ("The dog is named Luna", 0.3, 10, False), ("Exam on Friday about databases", 0.6, 1, True)])
        q = lambda **kw: self.get("/api/memory/list?" + "&".join("%s=%s" % kv for kv in kw.items()))[1]
        self.assertEqual([m["id"] for m in q(q="mango")["items"]], ["m0001"])
        self.assertEqual([m["id"] for m in q(q="MARIA")["items"]], ["m0001"])             # node names count, case does not
        self.assertEqual([m["id"] for m in q(q="m0002")["items"]], ["m0002"])
        self.assertEqual(q(q="mango+dog")["matched"], 0)                                  # every word must match
        self.assertEqual([m["id"] for m in q(pinned="1")["items"]], ["m0003"])
        self.assertEqual([m["id"] for m in q(sort="importance")["items"]], ["m0001", "m0003", "m0002"])
        self.assertEqual([m["id"] for m in q(sort="oldest")["items"]], ["m0002", "m0001", "m0003"])
        self.assertEqual([m["id"] for m in q(sort="newest")["items"]], ["m0003", "m0001", "m0002"])
        page = q(limit=1, offset=1)
        self.assertEqual((len(page["items"]), page["matched"]), (1, 3))

    def test_bad_parameters_are_refused(self):
        self.seed([("one two three", 0.5, 1, False)])
        for qs in ("status=nope", "sort=random", "limit=0", "limit=501", "limit=abc", "offset=-1", "q=" + "x" * 201):
            self.assertEqual(self.get("/api/memory/list?" + qs)[0], 400, qs)

    def test_a_broken_file_gives_a_sentence_without_the_folder(self):
        with open(self.mem_path, "w") as f:
            f.write("{not json")
        st, d = self.get("/api/memory/list")
        self.assertEqual(st, 400)
        self.assertNotIn(self.root, d["error"])

    def test_needs_token_host_origin(self):
        self.seed([("secret plan", 0.5, 1, False)])
        for kw in ({"token": False}, {"token": "wrong"}):
            st, _, body = self.api("GET", "/api/memory/list", **kw)
            self.assertEqual(st, 401); self.assertNotIn(b"secret plan", body)
        st, _, body = self.api("GET", "/api/memory/list", host="evil.example:80")
        self.assertEqual(st, 403); self.assertNotIn(b"secret plan", body)
        st, _, body = self.api("POST", "/api/memory/add", {"text": "x"}, headers={"Origin": "http://evil.example"})
        self.assertEqual(st, 403)
        st, _, body = self.api("POST", "/api/memory/add", {"text": "x"}, token=False)
        self.assertEqual(st, 401)
        self.assertEqual(len(self.disk()["memories"]), 1)


class AddEditForget(MemBase):
    def test_add_creates_the_file_and_logs_without_text(self):
        st, d = self.post("/api/memory/add", {"text": "I like [[Mango]] shake", "emotion": "joy", "importance": 0.7, "pinned": True})
        self.assertEqual(st, 200, d)
        m = d["memory"]
        self.assertEqual((m["id"], m["emotion"], m["importance"], m["pinned"], m["source"]), ("m0001", "joy", 0.7, True, "owner"))
        self.assertEqual(self.disk()["memories"]["m0001"]["text"], "I like Mango shake")
        with open(os.path.join(self.root, "memory.log.jsonl"), encoding="utf-8") as f:
            log = f.read()
        self.assertIn('"add"', log); self.assertNotIn("mango", log.lower())

    def test_add_validation_same_as_typed_commands(self):
        bad = ({}, {"text": ""}, {"text": "   "}, {"text": 5}, {"text": "x" * 2001}, {"text": "ok", "importance": 2}, {"text": "ok", "importance": "high"},
               {"text": "ok", "importance": True}, {"text": "ok", "emotion": "Two Words"}, {"text": "ok", "emotion": 3}, {"text": "ok", "pinned": "yes"})
        for b in bad:
            self.assertEqual(self.post("/api/memory/add", b)[0], 400, b)
        self.assertFalse(os.path.exists(self.mem_path))                      # nothing half-made
        self.assertEqual(self.post("/api/memory/add", {"text": "ok"})[0], 200)

    def test_edit_text_relinks_and_keeps_hand_made_links(self):
        s = self.seed([("[[person:Maria]] loves mango", 0.5, 1, False)])
        s = MEM.MemoryStore(self.mem_path)
        s.memories["m0001"]["nodes"].append(s.ensure_node("Cebu", "topic")["id"])      # a link made by hand (like CLI --node)
        s.save()
        st, d = self.post("/api/memory/edit", {"id": "m0001", "text": "[[person:Jun]] loves mango"})
        self.assertEqual(st, 200, d)
        names = sorted(n["name"] for n in d["memory"]["nodes"])
        self.assertEqual(names, ["Cebu", "Jun"])                              # Maria's link came from the old text, so it left; Cebu stayed
        self.assertNotIn("person:maria", self.disk()["nodes"])               # and the unused node is gone

    def test_edit_fields_and_pin(self):
        self.seed([("one two three", 0.5, 1, False)])
        st, d = self.post("/api/memory/edit", {"id": "m0001", "pinned": True})
        self.assertEqual((st, d["memory"]["pinned"], d["memory"]["importance"]), (200, True, 0.5))
        st, d = self.post("/api/memory/edit", {"id": "m0001", "importance": 0.9, "emotion": "stress"})
        self.assertEqual((d["memory"]["importance"], d["memory"]["emotion"], d["memory"]["pinned"]), (0.9, "stress", True))
        self.assertEqual(self.post("/api/memory/edit", {"id": "m0001", "pinned": False})[1]["memory"]["pinned"], False)

    def test_rejected_edit_changes_nothing(self):
        self.seed([("[[person:Maria]] loves mango", 0.5, 1, False)])
        before = self.disk()
        for b in ({"id": "m0001", "text": ""}, {"id": "m0001", "text": "x" * 2001}, {"id": "m0001", "importance": 3}, {"id": "m0001", "emotion": "A B"},
                  {"id": "m0001", "text": "[[person:New]] ok", "importance": 3}, {"id": "m0001"}, {"id": "m0001", "pinned": "x"},
                  {"id": "../x", "text": "a"}, {"id": 1, "text": "a"}, {"text": "a"}):
            self.assertEqual(self.post("/api/memory/edit", b)[0], 400, b)
        self.assertEqual(self.disk(), before)                                # even the "[[person:New]]" node was not left behind
        self.assertEqual(self.post("/api/memory/edit", {"id": "m0099", "text": "a"})[0], 404)

    def test_forget_needs_the_id_typed_back_and_leaves_no_backup(self):
        self.seed([("my secret note about luna", 0.5, 1, False), ("keep me", 0.5, 1, False)])
        self.post("/api/memory/edit", {"id": "m0001", "importance": 0.6})   # makes a .prev.json that still holds the text
        self.assertTrue(os.path.exists(os.path.join(self.root, "memory.prev.json")))
        for b in ({"id": "m0001"}, {"id": "m0001", "confirm": True}, {"id": "m0001", "confirm": "m0002"}, {"id": "m0001", "confirm": "1"}):
            self.assertEqual(self.post("/api/memory/forget", b)[0], 400, b)
        self.assertIn("m0001", self.disk()["memories"])
        st, d = self.post("/api/memory/forget", {"id": "m0001", "confirm": "m0001"})
        self.assertEqual((st, d["forgotten"]), (200, "m0001"))
        self.assertEqual(sorted(self.disk()["memories"]), ["m0002"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "memory.prev.json")))
        with open(os.path.join(self.root, "memory.log.jsonl"), encoding="utf-8") as f:
            self.assertNotIn("secret", f.read())
        self.assertEqual(self.post("/api/memory/forget", {"id": "m0001", "confirm": "m0001"})[0], 404)

    def test_a_file_changed_by_another_program_is_not_overwritten(self):
        self.seed([("one two three", 0.5, 1, False)])
        hub = self.state.memory
        orig = MEM.MemoryStore.save

        def clash(store, keep_backup=True):
            with open(self.mem_path, "a") as f:
                f.write(" ")                                                  # another program touches the file between load and save
            return orig(store, keep_backup)
        MEM.MemoryStore.save = clash
        try:
            st, d = self.post("/api/memory/add", {"text": "new thing here"})
        finally:
            MEM.MemoryStore.save = orig
        self.assertEqual(st, 409, d)
        self.assertNotIn(self.root, d["error"])


class Maintenance(MemBase):
    def setUp(self):
        super().setUp()
        self.seed([("I really love eating mango every summer", 0.6, 1, False), ("I really love eating mango every summer!", 0.5, 1, False),
                   ("old faded thing about rain", 0.2, 400, False), ("pinned old thing about snow", 0.2, 400, True), ("fresh note about school", 0.5, 1, False)])

    def test_plan_changes_nothing_and_shows_texts(self):
        before = self.disk()
        st, d = self.get("/api/memory/plan")
        self.assertEqual(st, 200, d)
        self.assertFalse(d["empty"])
        self.assertEqual([x["id"] for x in d["fade"]], ["m0003"])             # the pinned old one never fades
        self.assertEqual([(x["keep"], x["drop"]) for x in d["duplicates"]], [("m0001", "m0002")])
        self.assertIn("mango", d["duplicates"][0]["keep_text"])
        self.assertEqual(self.disk(), before)

    def test_apply_needs_the_matching_plan_id(self):
        for b in ({}, {"plan_id": ""}, {"plan_id": 5}, {"plan_id": "0123456789abcdef"}):
            st, d = self.post("/api/memory/maintain", b)
            self.assertIn(st, (400, 409), b)
        self.assertEqual({m["status"] for m in self.disk()["memories"].values()}, {"active"})
        pid = self.get("/api/memory/plan")[1]["plan_id"]
        st, d = self.post("/api/memory/maintain", {"plan_id": pid})
        self.assertEqual(st, 200, d)
        self.assertEqual((d["done"]["fade"], d["done"]["merged"]), (1, 1))
        mems = self.disk()["memories"]
        self.assertEqual((mems["m0003"]["status"], mems["m0002"]["status"], mems["m0004"]["status"]), ("dormant", "merged", "active"))
        self.assertEqual(len(mems), 5)                                        # nothing deleted
        self.assertEqual(self.post("/api/memory/maintain", {"plan_id": pid})[0], 400)    # nothing left to do now

    def test_a_stale_preview_is_not_applied(self):
        pid = self.get("/api/memory/plan")[1]["plan_id"]
        self.post("/api/memory/forget", {"id": "m0002", "confirm": "m0002"})
        st, d = self.post("/api/memory/maintain", {"plan_id": pid})
        self.assertIn(st, (400, 409), d)
        self.assertEqual(self.disk()["memories"]["m0003"]["status"], "active")

    def test_no_file_no_plan(self):
        os.remove(self.mem_path)
        self.assertEqual(self.get("/api/memory/plan")[0], 404)
        self.assertEqual(self.post("/api/memory/maintain", {"plan_id": "x"})[0], 404)


class Safety(MemBase):
    def test_only_memory_json_is_touched(self):
        self.seed([("one two three", 0.5, 1, False)])
        before = set(os.listdir(self.root))
        for b in ({"text": "hi", "path": "../x.json", "file": "data_msgr/private/raw_chat_notes.txt"},):
            self.post("/api/memory/add", dict(b))
        self.assertLessEqual(set(os.listdir(self.root)) - before, {"memory.log.jsonl", "memory.prev.json"})
        st, _, body = self.api("GET", "/api/memory/list?path=data_msgr/private/raw_chat_notes.txt")
        self.assertNotIn(SECRET.encode(), body)

    def test_a_memory_link_into_private_is_refused(self):
        if not self.links:
            self.skipTest("no symlinks here")
        os.remove(self.mem_path) if os.path.exists(self.mem_path) else None
        try:
            os.symlink(os.path.join(self.root, "data_msgr", "private", "raw_chat_notes.txt"), self.mem_path)
        except OSError:
            self.skipTest("no symlinks here")
        try:
            st, d = self.get("/api/memory/list")
            self.assertIn(st, (400, 403), d)
            self.assertNotIn(SECRET, json.dumps(d))
        finally:
            os.remove(self.mem_path)

    def test_chat_text_cannot_reach_the_memory_routes(self):
        self.seed([("one two three", 0.5, 1, False)])
        before = self.disk()
        for msg in ("/remember I owe you money", "/forget m0001", "POST /api/memory/forget", "/approve 1"):
            st, _, _ = self.api("POST", "/api/ping", {"message": msg})
            self.assertEqual(st, 200)
        self.assertEqual(self.disk(), before)

    def test_no_other_methods_and_no_extra_routes(self):
        for m in ("PUT", "DELETE", "PATCH"):
            self.assertEqual(self.api(m, "/api/memory/forget", {})[0], 405)
        for p in ("/api/memory", "/api/memory/delete", "/api/memory/export", "/api/memory/raw", "/api/memory/import"):
            for m in ("GET", "POST"):
                self.assertEqual(self.api(m, p, {} if m == "POST" else None)[0], 404, (m, p))
        self.assertEqual(self.api("POST", "/api/memory/list", {})[0], 404)         # reading is GET only
        self.assertEqual(self.api("GET", "/api/memory/add")[0], 404)               # changing is POST only

    def test_bad_bodies(self):
        self.assertEqual(self.api("POST", "/api/memory/add", "[1,2]")[0], 400)
        self.assertEqual(self.api("POST", "/api/memory/add", "{nope")[0], 400)
        self.assertEqual(self.api("POST", "/api/memory/add", "{}", headers={"Content-Type": "text/plain"})[0], 415)


if __name__ == "__main__":
    unittest.main()
