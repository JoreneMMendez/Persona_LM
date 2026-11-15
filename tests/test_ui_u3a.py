"""Stage U3a: the Prompt panel (GET /api/chat/prompt) and the opt-in memory / archive loading. Same tiny random model as test_ui_u1a."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import ui_server as UI
from persona_lm.archive import ChatArchive
from persona_lm.memory import MemoryStore
from test_ui_u0 import SECRET
from test_ui_u1a import ChatBase


class Prompt(ChatBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.remove(os.path.join(cls.root, "memory.json"))                    # the shared test folder starts with an empty stub
        store = MemoryStore(os.path.join(cls.root, "memory.json"))        # real memories, written before any model is loaded
        store.add("Luna loves mango and the dog", pinned=True)
        store.add("Maria is my sister, she loves mango")
        store.save() if hasattr(store, "save") else None
        arch = ChatArchive(os.path.join(cls.root, "chat_archive.jsonl"))
        arch.append("old1", "do you like mango", "yes i love mango so much", t=1.0)

    def setUp(self):
        self.post("/api/chat/reset") if self.get("/api/chat/state")[1].get("loaded") else None

    def test_needs_token_and_a_model(self):
        self.assertEqual(self.api("GET", "/api/chat/prompt", token=False)[0], 401)
        self.assertEqual(self.api("GET", "/api/chat/prompt", token="wrong")[0], 401)
        self.assertEqual(self.api("GET", "/api/chat/prompt", host="evil.example.com")[0], 403)
        self.state.chat.ai = None
        self.assertEqual(self.get("/api/chat/prompt")[0], 409)

    def test_no_prompt_before_the_first_message(self):
        self.load()
        st, d = self.get("/api/chat/prompt")
        self.assertEqual(st, 200)
        self.assertFalse(d["has_prompt"])
        self.assertFalse(d["memory_on"])

    def test_sections_are_the_exact_prompt(self):
        self.load(system="You are Luna.")
        self.say("hello how are you")
        st, d = self.get("/api/chat/prompt")
        self.assertEqual(st, 200)
        self.assertTrue(d["has_prompt"])
        kinds = [s["kind"] for s in d["sections"]]
        self.assertEqual(kinds, ["system", "user", "assistant"])               # the last one is the open "AI writes here" marker
        self.assertTrue(d["sections"][-1]["open"])
        self.assertEqual(d["sections"][0]["text"], "You are Luna.")
        self.assertEqual(d["sections"][1]["text"], "hello how are you")
        self.assertEqual(d["tokens"], len(self.state.chat.last_prompt["ids"]))   # block sizes add up to the real prompt length
        self.assertEqual(d["messages_in"], 1)
        self.assertEqual(d["messages_left_out"], 0)
        self.assertEqual(d["memories"], [])
        self.assertLessEqual(d["tokens"], d["budget"])

    def test_memory_opt_in_shows_recalled_memories(self):
        st = self.load(memory=True)
        self.assertTrue(st["memory_on"])
        self.say("do you know luna loves mango")
        d = self.get("/api/chat/prompt")[1]
        self.assertIn("memory", [s["kind"] for s in d["sections"]])
        self.assertTrue(d["memories"])
        self.assertTrue(any("mango" in m["text"].lower() for m in d["memories"]))
        mem = [s for s in d["sections"] if s["kind"] == "memory"][0]
        self.assertTrue(mem["text"].startswith("- "))                              # the lines the model really read
        self.assertIn(d["memories"][0]["text"].split()[0], mem["text"])

    def test_archive_opt_in_shows_old_chats_and_is_off_by_default(self):
        self.assertFalse(self.load()["archive_on"])
        st = self.load(archive=True)
        self.assertTrue(st["archive_on"])
        self.state.chat.ai.session.memory_tokens = 400                              # the tiny test tokenizer makes every line long
        self.say("do you like mango")
        d = self.get("/api/chat/prompt")[1]
        self.assertTrue(d["archive_on"])
        self.assertTrue(any("mango" in o["user"] for o in d["old_chats"]))
        self.assertIn("earlier chat", " ".join(s["text"] for s in d["sections"] if s["kind"] == "memory"))

    def test_old_messages_that_do_not_fit_are_counted(self):
        self.load(max_new=100)
        for _ in range(8):
            self.say("hello how are you today fine thanks and the cat sat on the mat " * 3)
        d = self.get("/api/chat/prompt")[1]
        self.assertGreater(d["messages_left_out"], 0)
        self.assertEqual(d["messages_in"] + d["messages_left_out"], len(self.get("/api/chat/state")[1]["history"]) - 1)   # the newest reply was not in the prompt

    def test_text_typed_as_a_command_is_only_a_message(self):
        self.load(memory=True)
        self.say("/forget all")
        d = self.get("/api/chat/prompt")[1]
        self.assertEqual([s for s in d["sections"] if s["kind"] == "user"][-1]["text"], "/forget all")
        self.assertEqual(len(self.get("/api/chat/state")[1]["history"]), 2)
        with open(os.path.join(self.root, "memory.json"), encoding="utf-8") as f:
            self.assertIn("Luna loves mango", f.read())                             # "/forget all" did nothing

    def test_reset_and_reload_forget_the_old_prompt(self):
        self.load()
        self.say("hello")
        self.post("/api/chat/reset")
        self.assertFalse(self.get("/api/chat/prompt")[1]["has_prompt"])
        self.say("hello")
        self.load()
        self.assertFalse(self.get("/api/chat/prompt")[1]["has_prompt"])

    def test_opt_in_refusals(self):
        os.rename(os.path.join(self.root, "memory.json"), os.path.join(self.root, "memory.hold"))
        try:
            st, d = self.post("/api/chat/load", dict(self.good, memory=True))
            self.assertEqual(st, 400)
            self.assertIn("memory.json", d["error"])
        finally:
            os.rename(os.path.join(self.root, "memory.hold"), os.path.join(self.root, "memory.json"))
        for bad in ("yes", 1, "../outside/x"):                                      # only the literal true switches it on; no path is ever taken from the page
            st, d = self.post("/api/chat/load", dict(self.good, memory=bad, archive=bad))
            self.assertEqual(st, 200, d)
            self.assertFalse(d["memory_on"])
            self.assertFalse(d["archive_on"])

    def test_nothing_private_or_absolute_in_the_answer(self):
        self.load(memory=True, archive=True)
        self.say("hello")
        raw = self.api("GET", "/api/chat/prompt")[2]
        self.assertNotIn(SECRET.encode(), raw)
        self.assertNotIn(self.root.encode(), raw)

    def test_route_is_get_only(self):
        self.assertIn("/api/chat/prompt", UI.CHAT_GET)
        self.assertNotIn("/api/chat/prompt", UI.CHAT_POST)
        self.assertEqual(self.api("POST", "/api/chat/prompt", {})[0], 404)


if __name__ == "__main__":
    unittest.main()
