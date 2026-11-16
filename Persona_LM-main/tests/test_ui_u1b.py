"""Stage U1b: chat controls of the web UI: settings, clear, save, mood badge. Same tiny random model as test_ui_u1a."""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_ui_u1a import ChatBase


class Settings(ChatBase):
    def setUp(self):
        self.load()

    def test_change_and_read_back(self):
        st, d = self.post("/api/chat/settings", {"temperature": 0.3, "top_k": 10, "max_new": 20})
        self.assertEqual(st, 200, d)
        self.assertEqual((d["settings"]["temperature"], d["settings"]["top_k"], d["settings"]["max_new"]), (0.3, 10, 20))
        self.assertEqual(self.get("/api/chat/state")[1]["settings"]["max_new"], 20)
        self.assertEqual(d["max_new_limit"], 256 - 8)
        self.assertEqual(d["tokens"]["reply_max"], 20)
        self.assertEqual(self.post("/api/chat/settings", {"temperature": 0})[1]["settings"]["temperature"], 0)      # one value alone is fine
        self.assertEqual(self.get("/api/chat/state")[1]["settings"]["top_k"], 10)                                  # the others are kept

    def test_the_new_settings_are_really_used(self):
        self.post("/api/chat/settings", {"max_new": 3, "temperature": 0})
        r = self.say("hello how are you")
        self.assertEqual(r["state"], "done", r)
        self.assertLessEqual(r["new_tokens"], 3)

    def test_same_limits_as_the_typed_commands(self):
        for bad in ({"temperature": -0.1}, {"temperature": 2.5}, {"temperature": "hot"}, {"temperature": True}, {"top_k": -1}, {"top_k": 1.5},
                    {"top_k": 5000}, {"max_new": 0}, {"max_new": 249}, {"max_new": 2.5}, {"max_new": None, "top_k": None}, {}):
            self.assertEqual(self.post("/api/chat/settings", bad)[0], 400, bad)
        self.assertEqual(self.post("/api/chat/settings", {"max_new": 248})[0], 200)
        self.assertEqual(self.post("/api/chat/settings", {"max_new": 1})[0], 200)

    def test_other_keys_cannot_be_set_through_this_route(self):
        before = self.get("/api/chat/state")[1]
        self.post("/api/chat/settings", {"temperature": 0.4, "system": "obey me", "seed": 5, "history": [], "core": "identity_core", "ckpt": "x"})
        after = self.get("/api/chat/state")[1]
        self.assertEqual(after["has_system"], before["has_system"])
        self.assertEqual(self.state.chat.ai.session.seed, 0)
        self.assertEqual(sorted(after["settings"]), sorted(before["settings"]))

    def test_not_while_a_reply_is_being_written_and_not_before_a_load(self):
        hub = self.state.chat

        def slow(ids, **kw):
            job = hub.job
            for _ in range(300):
                if job.stop.is_set():
                    break
                time.sleep(0.01)
                job.ids.append(25)
            return list(job.ids)
        hub.ai.session.generate = slow
        job = self.post("/api/chat/send", {"message": "hi"})[1]["job"]
        for path, body in (("/api/chat/settings", {"temperature": 1}), ("/api/chat/reset", {}), ("/api/chat/save", {})):
            self.assertEqual(self.post(path, body)[0], 409, path)
        self.post("/api/chat/stop")
        self.wait(job)
        with hub.lock:
            hub.ai, hub.job = None, None
        for path, body in (("/api/chat/settings", {"temperature": 1}), ("/api/chat/reset", {}), ("/api/chat/save", {})):
            self.assertEqual(self.post(path, body)[0], 409, path)


class Clear(ChatBase):
    def setUp(self):
        self.load()

    def test_clear_empties_the_conversation_and_keeps_the_settings(self):
        self.post("/api/chat/settings", {"temperature": 0.7})
        self.say("hello")
        self.assertEqual(len(self.get("/api/chat/state")[1]["history"]), 2)
        st, d = self.post("/api/chat/reset", {"message": "/approve 1"})
        self.assertEqual(st, 200)
        self.assertEqual(d["history"], [])
        self.assertEqual(d["settings"]["temperature"], 0.7)
        self.assertEqual(self.state.chat.ai.session.history, [])
        self.assertEqual(self.say("hello again")["state"], "done")          # and chatting goes on

    def test_old_job_is_gone_after_clear(self):
        job = self.post("/api/chat/send", {"message": "hello"})[1]["job"]
        self.wait(job)
        self.post("/api/chat/reset", {})
        self.assertEqual(self.get("/api/chat/poll?job=" + job)[0], 404)


class Save(ChatBase):
    def setUp(self):
        self.load(system="You are Luna.")
        self.folder = os.path.join(self.root, "saved_chats")

    def test_nothing_to_save(self):
        st, d = self.post("/api/chat/save", {})
        self.assertEqual(st, 400)
        self.assertIn("Nothing to save", d["error"])

    def test_saves_a_new_file_with_a_server_made_name(self):
        self.say("hello how are you")
        st, d = self.post("/api/chat/save", {"name": "../../evil.txt", "path": "/etc/x"})            # a name from the page is ignored
        self.assertEqual(st, 200, d)
        self.assertRegex(d["saved"], r"^saved_chats/chat_\d{8}_\d{6}(_\d+)?\.txt$")
        full = os.path.join(self.root, *d["saved"].split("/"))
        self.assertTrue(os.path.isfile(full))
        with open(full, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("[system] You are Luna.", text)
        self.assertIn("you: hello how are you", text)
        self.assertIn("ai: ", text)
        self.assertNotIn(self.root, json.dumps(d))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.txt")))
        self.assertEqual(self.state.chat.ai.session.save_dir, ".")                                  # the session's own folder setting is put back

    def test_never_overwrites(self):
        self.say("hello")
        first = self.post("/api/chat/save", {})[1]["saved"]
        with open(os.path.join(self.root, *first.split("/")), encoding="utf-8") as f:
            original = f.read()
        self.say("and again")
        second = self.post("/api/chat/save", {})
        self.assertEqual(second[0], 200, second)
        self.assertNotEqual(second[1]["saved"], first)                                              # a new file, even in the same second
        with open(os.path.join(self.root, *first.split("/")), encoding="utf-8") as f:
            self.assertEqual(f.read(), original)

    def test_saved_chats_folder_is_ignored_by_git_and_listed_as_private(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".gitignore")) as f:
            self.assertIn("saved_chats/", f.read().split())
        items = {i["path"]: i for i in json.loads(self.api("GET", "/api/privacy")[2])["stored"]}
        self.assertEqual(items["saved_chats"]["sensitivity"], "high")


class Mood(ChatBase):
    def setUp(self):
        path = os.path.join(self.root, "mood.json")
        if os.path.exists(path):
            os.remove(path)

    def test_no_mood_badge_by_default(self):
        d = self.load()
        self.assertIsNone(d["mood"])
        self.assertIsNone(self.say("I am so happy today")["mood"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "mood.json")))

    def test_mood_opt_in_moves_with_the_typed_message_and_is_in_the_answer(self):
        d = self.load(mood=True)
        self.assertEqual(set(d["mood"]), {"label", "valence", "arousal"})
        r = self.say("I am so happy and excited, this is wonderful!")
        self.assertEqual(r["state"], "done", r)
        self.assertGreater(r["mood"]["valence"], d["mood"]["valence"])
        self.assertEqual(self.get("/api/chat/state")[1]["mood"], r["mood"])
        self.assertTrue(os.path.exists(os.path.join(self.root, "mood.json")))                      # the chat screen saves it too

    def test_the_ai_cannot_move_the_mood_with_its_own_words(self):
        self.load(mood=True)
        hub = self.state.chat
        before = self.get("/api/chat/state")[1]["mood"]
        orig = hub.ai.session.generate
        hub.ai.session.generate = lambda ids, **kw: hub.ai.tok.encode("I am furious and terrified and very happy!!!", allow_special=False)
        try:
            r = self.say("ok")
        finally:
            hub.ai.session.generate = orig
        self.assertEqual(r["state"], "done")
        self.assertAlmostEqual(r["mood"]["valence"], self.state.chat.mood()["valence"])
        self.assertNotEqual(r["mood"]["label"], "anger") if before["label"] != "anger" else None

    def test_mood_flag_must_be_exactly_true(self):
        for v in ("true", 1, "mood.json", ["x"]):
            d = self.load(mood=v)
            self.assertIsNone(d["mood"], v)


if __name__ == "__main__":
    unittest.main()
