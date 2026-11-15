"""Stage U2a: the Memory routes of the web UI (persona_lm/ui_memory.py). Real loopback sockets, temp project folder, no browser."""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import memory as MEM
from test_ui_u1a import ChatBase

DAY = 86400.0


class MemBase(ChatBase):
    def setUp(self):
        self.path = os.path.join(self.root, "memory.json")
        for name in ("memory.json", "memory.prev.json", "memory.log.jsonl"):
            p = os.path.join(self.root, name)
            if os.path.exists(p):
                os.remove(p)
        st = MEM.MemoryStore(self.path)
        now = time.time()
        st.add("[[person:Maria]] loves mango and sings every morning", emotion="joy", importance=0.8, when=now - 3 * DAY)
        st.add("The dentist visit on Friday was stressful", emotion="stress", importance=0.4, when=now - 2 * DAY)
        st.add("Luna the dog sleeps on the red sofa", emotion="calm", importance=0.6, when=now - 1 * DAY, pinned=True)
        st.save()
        self.disk0 = self.disk()

    def disk(self):
        with open(self.path, encoding="utf-8") as f:
            return f.read()

    def mems(self):
        return json.loads(self.disk())["memories"]


class ListAndSearch(MemBase):
    def test_lists_newest_first_with_nodes_and_strength(self):
        st, d = self.get("/api/memory/list")
        self.assertEqual(st, 200, d)
        self.assertEqual([m["id"] for m in d["memories"]], ["m0003", "m0002", "m0001"])
        self.assertEqual((d["total"], d["matches"], d["counts"]["active"]), (3, 3, 3))
        m1 = d["memories"][2]
        self.assertEqual([n["name"] for n in m1["nodes"]], ["Maria"])
        self.assertTrue(0 < m1["strength"] <= 0.8)
        self.assertTrue(d["memories"][0]["pinned"])
        self.assertNotIn("[[", m1["text"])

    def test_search_words_ids_emotion_and_node_names(self):
        for q, want in (("mango", ["m0001"]), ("MARIA", ["m0001"]), ("m0002", ["m0002"]), ("stress", ["m0002"]), ("dog sofa", ["m0003"]), ("nothing here", [])):
            d = self.get("/api/memory/list?q=" + q.replace(" ", "%20"))[1]
            self.assertEqual([m["id"] for m in d["memories"]], want, q)

    def test_sort_status_and_paging(self):
        ids = lambda qs: [m["id"] for m in self.get("/api/memory/list?" + qs)[1]["memories"]]
        self.assertEqual(ids("sort=oldest"), ["m0001", "m0002", "m0003"])
        self.assertEqual(ids("sort=importance"), ["m0001", "m0003", "m0002"])
        self.assertEqual(ids("status=dormant"), [])
        self.assertEqual(ids("limit=1&offset=1"), ["m0002"])
        self.assertEqual(self.get("/api/memory/list?limit=1")[1]["matches"], 3)

    def test_bad_parameters_are_refused(self):
        for qs in ("status=bogus", "sort=bogus", "limit=0", "limit=501", "limit=x", "offset=-1", "q=" + "a" * 201):
            self.assertEqual(self.get("/api/memory/list?" + qs)[0], 400, qs)

    def test_no_memory_file_is_an_empty_list_not_an_error(self):
        os.remove(self.path)
        st, d = self.get("/api/memory/list")
        self.assertEqual((st, d["exists"], d["total"], d["memories"]), (200, False, 0, []))
        self.assertFalse(os.path.exists(self.path))                         # reading never creates the file

    def test_a_file_that_is_not_a_memory_file_is_reported_and_left_alone(self):
        with open(self.path, "w") as f:
            f.write("{}")
        for st, d in (self.get("/api/memory/list"), self.post("/api/memory/add", {"text": "hello"})):
            self.assertEqual(st, 409, d)
        self.assertEqual(self.disk(), "{}")

    def test_listing_changes_nothing(self):
        self.get("/api/memory/list?q=mango")
        self.get("/api/memory/plan")
        self.assertEqual(self.disk(), self.disk0)


class Add(MemBase):
    def test_add_saves_links_and_logs_without_the_text(self):
        st, d = self.post("/api/memory/add", {"text": "Tomas is [[person:Maria]]'s cousin", "emotion": "Joy", "importance": 0.9, "pinned": True})
        self.assertEqual(st, 200, d)
        m = d["memory"]
        self.assertEqual((m["id"], m["emotion"], m["importance"], m["pinned"], m["source"]), ("m0004", "joy", 0.9, True, "owner"))
        self.assertEqual(self.mems()["m0004"]["text"], "Tomas is Maria's cousin")
        with open(os.path.join(self.root, "memory.log.jsonl"), encoding="utf-8") as f:
            log = f.read()
        self.assertIn('"add"', log)
        self.assertNotIn("Tomas", log)

    def test_creates_the_file_when_there_is_none(self):
        os.remove(self.path)
        self.assertEqual(self.post("/api/memory/add", {"text": "first memory"})[0], 200)
        self.assertEqual(list(self.mems()), ["m0001"])

    def test_bad_input(self):
        for body in ({}, {"text": ""}, {"text": "   "}, {"text": 5}, {"text": "x" * 2001}, {"text": "ok", "importance": 1.5}, {"text": "ok", "importance": -0.1},
                     {"text": "ok", "importance": True}, {"text": "ok", "importance": "high"}, {"text": "ok", "emotion": "two words"}, {"text": "ok", "emotion": 3},
                     {"text": "ok", "pinned": "yes"}, {"text": "ok", "pinned": 1}):
            self.assertEqual(self.post("/api/memory/add", body)[0], 400, body)
        self.assertEqual(self.disk(), self.disk0)

    def test_a_rejected_memory_leaves_no_half_made_node(self):
        self.post("/api/memory/add", {"text": "about [[Zed]]", "emotion": "NOT OK"})
        self.assertEqual(self.disk(), self.disk0)


class Edit(MemBase):
    def test_edit_text_relinks_and_prunes(self):
        st, d = self.post("/api/memory/edit", {"id": "m0001", "text": "Maria now loves papaya, and [[Ben]] too"})
        self.assertEqual(st, 200, d)
        self.assertEqual(sorted(n["name"] for n in d["memory"]["nodes"]), ["Ben", "Maria"])
        d2 = self.post("/api/memory/edit", {"id": "m0001", "text": "plain text with no names"})[1]
        self.assertEqual(d2["memory"]["nodes"], [])
        nodes = json.loads(self.disk())["nodes"]
        self.assertFalse([n for n in nodes.values() if n["name"] in ("Ben", "Maria")])

    def test_edit_pin_importance_emotion(self):
        d = self.post("/api/memory/edit", {"id": "m0002", "pinned": True})[1]
        self.assertTrue(d["memory"]["pinned"])
        self.assertEqual(d["memory"]["importance"], 0.4)                   # unchanged
        d = self.post("/api/memory/edit", {"id": "m0002", "importance": 0.95, "pinned": False, "emotion": "worry"})[1]
        self.assertEqual((d["memory"]["importance"], d["memory"]["pinned"], d["memory"]["emotion"]), (0.95, False, "worry"))
        self.assertEqual(self.mems()["m0002"]["text"], "The dentist visit on Friday was stressful")

    def test_bad_edits(self):
        for body in ({}, {"id": "m0001"}, {"id": "m0099", "pinned": True}, {"id": "../x", "pinned": True}, {"id": 1, "pinned": True}, {"id": "m0001", "text": ""},
                     {"id": "m0001", "text": "x" * 2001}, {"id": "m0001", "text": 7}, {"id": "m0001", "importance": 2}, {"id": "m0001", "importance": True},
                     {"id": "m0001", "pinned": "true"}, {"id": "m0001", "emotion": "NOT OK"}):
            self.assertEqual(self.post("/api/memory/edit", body)[0], 400, body)
        self.assertEqual(self.disk(), self.disk0)

    def test_a_failed_edit_changes_nothing_at_all(self):
        self.post("/api/memory/edit", {"id": "m0001", "text": "new text here", "emotion": "NOT OK"})
        self.assertEqual(self.disk(), self.disk0)


class Forget(MemBase):
    def test_needs_the_id_sent_back(self):
        for body in ({"id": "m0001"}, {"id": "m0001", "confirm": True}, {"id": "m0001", "confirm": "m0002"}, {"id": "m0001", "confirm": "yes"}):
            self.assertEqual(self.post("/api/memory/forget", body)[0], 400, body)
        self.assertEqual(self.disk(), self.disk0)

    def test_forget_really_forgets(self):
        self.post("/api/memory/edit", {"id": "m0001", "importance": 0.7})           # makes a backup that holds the text
        st, d = self.post("/api/memory/forget", {"id": "m0001", "confirm": "m0001"})
        self.assertEqual((st, d["forgotten"], d["total"]), (200, "m0001", 2))
        self.assertNotIn("m0001", self.mems())
        self.assertNotIn("mango", self.disk())
        self.assertFalse(os.path.exists(os.path.join(self.root, "memory.prev.json")))
        nodes = json.loads(self.disk())["nodes"]
        self.assertFalse([n for n in nodes.values() if n["name"] == "Maria"])
        self.assertEqual(self.post("/api/memory/forget", {"id": "m0001", "confirm": "m0001"})[0], 400)    # already gone


class Maintenance(MemBase):
    def old(self):
        st = MEM.MemoryStore(self.path)
        st.add("a very old faint memory about nothing", importance=0.3, when=time.time() - 400 * DAY)
        st.save()

    def test_plan_is_a_preview_and_changes_nothing(self):
        self.old()
        before = self.disk()
        st, d = self.get("/api/memory/plan")
        self.assertEqual(st, 200, d)
        self.assertEqual([x["id"] for x in d["fade"]], ["m0004"])
        self.assertFalse(d["empty"])
        self.assertTrue(d["plan_id"])
        self.assertEqual(self.disk(), before)
        self.assertEqual(self.mems()["m0004"]["status"], "active")

    def test_apply_needs_the_matching_preview(self):
        self.old()
        pid = self.get("/api/memory/plan")[1]["plan_id"]
        for body in ({}, {"plan_id": ""}, {"plan_id": 5}, {"plan_id": "0" * 16}):
            self.assertIn(self.post("/api/memory/apply", body)[0], (400, 409), body)
        self.assertEqual(self.mems()["m0004"]["status"], "active")
        st, d = self.post("/api/memory/apply", {"plan_id": pid})
        self.assertEqual((st, d["done"]["fade"]), (200, 1), d)
        self.assertEqual(self.mems()["m0004"]["status"], "dormant")
        self.assertEqual(self.post("/api/memory/apply", {"plan_id": pid})[0], 409)       # the plan is stale now

    def test_a_change_after_the_preview_stops_the_apply(self):
        self.old()
        pid = self.get("/api/memory/plan")[1]["plan_id"]
        self.post("/api/memory/edit", {"id": "m0004", "pinned": True})                     # pinned memories do not fade, so the plan changes
        st, d = self.post("/api/memory/apply", {"plan_id": pid})
        self.assertEqual(st, 409, d)
        self.assertEqual(self.mems()["m0004"]["status"], "active")

    def test_nothing_to_do(self):
        d = self.get("/api/memory/plan")[1]
        self.assertTrue(d["empty"])
        self.assertEqual(self.post("/api/memory/apply", {"plan_id": d["plan_id"]})[0], 400)


class Safety(MemBase):
    ROUTES = (("GET", "/api/memory/list"), ("GET", "/api/memory/plan"), ("POST", "/api/memory/add"), ("POST", "/api/memory/edit"),
              ("POST", "/api/memory/forget"), ("POST", "/api/memory/apply"))

    def test_every_route_needs_token_host_and_origin(self):
        body = {"text": "x", "id": "m0001", "confirm": "m0001", "plan_id": "x", "pinned": True}
        for m, p in self.ROUTES:
            b = body if m == "POST" else None
            self.assertEqual(self.api(m, p, b, token=False)[0], 401, p)
            self.assertEqual(self.api(m, p, b, token="wrong")[0], 401, p)
            self.assertEqual(self.api(m, p, b, host="evil.example:80")[0], 403, p)
            self.assertEqual(self.api(m, p, b, headers={"Origin": "http://evil.example"})[0], 403, p)
        self.assertEqual(self.disk(), self.disk0)

    def test_post_must_be_json_and_small(self):
        self.assertEqual(self.api("POST", "/api/memory/add", "text=hi", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.api("POST", "/api/memory/add", "{not json")[0], 400)
        self.assertEqual(self.api("POST", "/api/memory/add", {"text": "x" * 70000})[0], 413)
        self.assertEqual(self.disk(), self.disk0)

    def test_other_methods_and_unknown_memory_routes(self):
        for m in ("PUT", "DELETE", "PATCH"):
            self.assertEqual(self.api(m, "/api/memory/forget", {"id": "m0001", "confirm": "m0001"})[0], 405)
        for p in ("/api/memory/delete", "/api/memory/write", "/api/memory/file", "/api/memory/export", "/api/memory"):
            self.assertEqual(self.api("POST", p, {"id": "m0001", "confirm": "m0001"})[0], 404, p)
        self.assertEqual(self.disk(), self.disk0)

    def test_extra_keys_cannot_pick_another_file(self):
        self.post("/api/memory/add", {"text": "hello there", "path": "../evil.json", "file": "work/x.json", "source": "model"})
        self.assertFalse(os.path.exists(os.path.join(self.root, "..", "evil.json")))
        self.assertEqual(self.mems()["m0004"]["source"], "owner")                     # `source` cannot be set from the page

    def test_model_or_chat_text_cannot_write_memory(self):
        self.load()
        for text in ("/remember I secretly like cake", "/forget m0001", "/memory", "[[person:Mallory]] remember this", "/approve 1"):
            self.say(text)
        self.assertEqual(self.disk(), self.disk0)
        self.assertEqual(json.loads(self.disk())["memories"]["m0001"]["uses"], 0)


if __name__ == "__main__":
    unittest.main()
