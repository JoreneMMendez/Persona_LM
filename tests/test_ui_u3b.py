"""Stage U3b: the personas page routes (GET /api/personas, POST /api/personas/use) and the persona fields in the chat state."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import identity as I
from persona_lm import personas as P
from persona_lm import ui_server as UI
from test_ui_u1a import ChatBase


class Personas(ChatBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        r = cls.root
        tok_path = os.path.join(r, "work", "tok.json")
        for n in ("mira", "kai"):
            core = I.Core.create(os.path.join(r, "identity_core", n), n.title(), "Jaymar", tokenizer=tok_path)
            core.add_version("v1", os.path.join(r, "work3", "run_chat2", "ckpt.npz"))
        ps = P.Personas(I.Core(os.path.join(r, "identity_core", "mira")))
        ps.add("calm mentor", "Calm.")
        ps.add("joker", "Joke.")
        ps.add("wordy", "You speak slowly and kindly and you always ask good questions and you never hurry. " * 3)
        cls.core = {"core": "identity_core/mira", "max_new": 8, "version": "v1"}

    def setUp(self):
        self.state.chat.ai = None
        P.Personas(I.Core(os.path.join(self.root, "identity_core", "mira"))).use("none")

    def test_needs_token_and_host(self):
        self.assertEqual(self.api("GET", "/api/personas", token=False)[0], 401)
        self.assertEqual(self.api("GET", "/api/personas", token="wrong")[0], 401)
        self.assertEqual(self.api("GET", "/api/personas", host="evil.example.com")[0], 403)
        self.assertEqual(self.api("POST", "/api/personas/use", {"name": "joker"}, token="wrong")[0], 401)
        self.assertEqual(self.api("POST", "/api/personas/use", {"name": "joker"}, host="evil.example.com")[0], 403)

    def test_nothing_loaded_lists_the_cores_only(self):
        st, d = self.get("/api/personas")
        self.assertEqual(st, 200)
        self.assertFalse(d["loaded"])
        self.assertEqual(sorted(c["path"] for c in d["cores"]), ["identity_core/kai", "identity_core/mira"])

    def test_read_only_listing_of_a_core_that_is_not_loaded(self):
        st, d = self.get("/api/personas?core=identity_core/mira")
        self.assertEqual(st, 200)
        self.assertFalse(d["switchable"])
        self.assertEqual([p["name"] for p in d["personas"]], ["calm mentor", "joker", "wordy"])
        self.assertIsNone(d["active"])
        self.assertEqual(self.get("/api/personas?core=identity_core/kai")[1]["personas"], [])
        self.assertEqual(self.post("/api/personas/use", {"name": "joker"})[0], 409)                 # nothing loaded: no switch

    def test_loaded_core_is_switchable_and_the_prompt_changes(self):
        self.post("/api/chat/load", self.core)
        self.state.chat.ai.session.system = None                                                    # the tiny test tokenizer makes the fixed mask sentence long
        d = self.get("/api/personas")[1]
        self.assertTrue(d["loaded"] and d["switchable"])
        self.assertIsNone(d["block"])
        st, r = self.post("/api/personas/use", {"name": "Joker"})                                  # same case-insensitive match as /persona
        self.assertEqual(st, 200, r)
        self.assertEqual(r["active"], "joker")
        self.assertIn("Joke.", r["block"])
        self.assertEqual(r["chat"]["persona"], "joker")
        self.say("hello")
        kinds = [s["kind"] for s in self.get("/api/chat/prompt")[1]["sections"]]
        self.assertIn("persona", kinds)
        st, r = self.post("/api/personas/use", {"name": "none"})
        self.assertEqual((st, r["active"]), (200, None))
        self.say("hello again")
        self.assertNotIn("persona", [s["kind"] for s in self.get("/api/chat/prompt")[1]["sections"]])

    def test_a_mask_too_long_for_the_window_is_refused_when_you_send(self):
        self.post("/api/chat/load", self.core)
        self.assertEqual(self.post("/api/personas/use", {"name": "wordy"})[0], 200)
        d = self.say("hello")
        self.assertEqual(d["state"], "error")
        self.assertIn("persona text is too long", d["error"])
        self.assertEqual(self.get("/api/chat/state")[1]["history"], [])                              # nothing was kept

    def test_switch_is_logged_and_kept_on_disk(self):
        self.post("/api/chat/load", self.core)
        self.post("/api/personas/use", {"name": "calm mentor"})
        self.assertEqual(P.Personas(I.Core(os.path.join(self.root, "identity_core", "mira"))).active, "calm mentor")

    def test_refusals(self):
        self.post("/api/chat/load", self.core)
        for bad in ("nobody", "", "x" * 200, None, 5, ["joker"]):
            self.assertEqual(self.post("/api/personas/use", {"name": bad})[0], 400, bad)
        self.assertEqual(self.get("/api/personas")[1]["active"], None)
        for core in ("identity_core/nope", "../outside", "work", "data_msgr/private"):
            self.assertIn(self.get("/api/personas?core=" + core)[0], (400,), core)

    def test_checkpoint_mode_has_no_personas(self):
        self.load()
        d = self.get("/api/personas")[1]
        self.assertFalse(d["loaded"])
        self.assertTrue(d["has_model"])
        self.assertEqual(self.post("/api/personas/use", {"name": "joker"})[0], 409)
        self.assertIsNone(self.get("/api/chat/state")[1]["personas"])

    def test_not_while_a_reply_is_being_written(self):
        self.post("/api/chat/load", self.core)
        hub = self.state.chat
        hub.send("hello")
        hub.job.state = "running"
        try:
            self.assertEqual(self.post("/api/personas/use", {"name": "joker"})[0], 409)
        finally:
            hub.job.state = "done"

    def test_typing_a_switch_in_the_chat_does_nothing(self):
        self.post("/api/chat/load", self.core)
        for text in ("/persona joker", "/persona none", "persona use joker"):
            self.say(text)
        self.assertIsNone(self.get("/api/personas")[1]["active"])
        self.assertEqual(self.get("/api/chat/state")[1]["persona"], None)

    def test_no_absolute_paths_in_answers(self):
        self.post("/api/chat/load", self.core)
        raw = self.api("GET", "/api/personas")[2] + self.api("GET", "/api/personas?core=identity_core/kai")[2]
        self.assertNotIn(self.root.encode(), raw)

    def test_route_tables(self):
        self.assertEqual(UI.PERSONA_GET, ["/api/personas"])
        self.assertEqual(UI.PERSONA_POST, ["/api/personas/use"])
        self.assertEqual(self.api("POST", "/api/personas", {})[0], 404)
        self.assertEqual(self.api("GET", "/api/personas/use")[0], 404)


if __name__ == "__main__":
    unittest.main()
