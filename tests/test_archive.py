"""Tests for the chat archive: old chats are kept in a file and the matching ones are shown to the AI (even from 200 chats ago).
Nothing here reads the keyboard or uses the network."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import archive as A
from persona_lm import chatrepl
from persona_lm.bpe import MEMORY, Tokenizer

DAY = 86400.0
T0 = 1_700_000_000.0


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, days):
        self.t += days * DAY


class FakeGen:
    def __init__(self, tok, text="sure thing"):
        self.tok, self.text, self.calls = tok, text, []

    def __call__(self, ids, **kw):
        self.calls.append(list(ids))
        return self.tok.encode(self.text, allow_special=False)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.path = os.path.join(self.root, "chat_archive.jsonl")
        self.clock = Clock()


class ArchiveFileTests(Base):
    def test_append_then_reload_keeps_everything_in_order(self):
        a = A.ChatArchive(self.path, clock=self.clock)
        a.append("s1", "hello there", "hi")
        a.append("s1", "how are you", "fine")
        b = A.ChatArchive(self.path, clock=self.clock)
        self.assertEqual([(i["id"], i["user"]) for i in b.items], [(1, "hello there"), (2, "how are you")])
        self.assertEqual(b.append("s2", "again", "yes")["id"], 3)

    def test_a_half_written_last_line_is_ignored_and_appending_still_works(self):
        a = A.ChatArchive(self.path, clock=self.clock)
        a.append("s1", "one", "uno")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"id": 2, "session": "s1", "t": 1, "user": "tw')          # power cut in the middle of a line
        b = A.ChatArchive(self.path, clock=self.clock)
        self.assertEqual(len(b), 1)

    def test_text_is_never_changed_except_long_messages_are_cut(self):
        a = A.ChatArchive(self.path, clock=self.clock)
        it = a.append("s1", "x" * 5000, "ok")
        self.assertLessEqual(len(it["user"]), A.MAX_TEXT)
        self.assertEqual(it["ai"], "ok")

    def test_unwritable_archive_gives_a_message_not_a_crash(self):
        a = A.ChatArchive(os.path.join(self.root, "nope", "x.jsonl"))
        os.makedirs(os.path.join(self.root, "nope"))
        os.rmdir(os.path.join(self.root, "nope"))
        with open(os.path.join(self.root, "nope"), "w") as f:                   # a FILE where the folder should be
            f.write("x")
        with self.assertRaises(A.ArchiveProblem):
            a.append("s1", "a", "b")


class SearchTests(Base):
    def test_rarer_words_win_and_no_common_word_means_no_result(self):
        a = A.ChatArchive(self.path, clock=self.clock)
        for i in range(30):
            a.append("s%d" % i, "what is up today", "nothing much today")
        a.append("s99", "my sister Maria loves mango", "that is sweet")
        got = a.search("do you remember Maria and the mango", k=3)
        self.assertEqual(got[0][0]["session"], "s99")
        self.assertEqual(a.search("quantum chromodynamics", k=3), [])

    def test_a_newer_exchange_beats_an_equal_old_one(self):
        a = A.ChatArchive(self.path, clock=self.clock)
        a.append("old", "i adopted a puppy", "congrats", t=T0 - 900 * DAY)
        a.append("new", "i adopted a puppy", "congrats", t=T0 - 2 * DAY)
        self.assertEqual(a.search("puppy", k=2)[0][0]["session"], "new")
        self.assertEqual(len(a.search("puppy", k=2)), 2)            # the old one is still found, just ranked lower

    def test_skip_ids_hides_exchanges_the_model_can_already_see(self):
        a = A.ChatArchive(self.path, clock=self.clock)
        first = a.append("s1", "i like durian", "ew")
        a.append("s2", "durian is great", "agreed")
        got = a.search("durian", k=5, skip_ids=[first["id"]])
        self.assertEqual([it["id"] for it, _, _ in got], [2])

    def test_when_text(self):
        now = T0
        self.assertEqual(A.when_text(now - 100, now), "today")
        self.assertEqual(A.when_text(now - 3 * DAY, now), "3 days ago")
        self.assertEqual(A.when_text(now - 95 * DAY, now), "3 months ago")
        self.assertEqual(A.when_text(now - 800 * DAY, now), "2 years ago")


class ChatTests(Base):
    def setUp(self):
        super().setUp()
        corpus = "the cat sat on the mat. maria loves mango and the dog. hello how are you today fine thanks earlier chat you said I said " * 40
        self.tok = Tokenizer.train([corpus], 300)

    def session(self, ctx=512, text="sure thing", **kw):
        self.gen = FakeGen(self.tok, text)
        self.arc = A.ChatArchive(self.path, clock=self.clock)
        return chatrepl.ChatSession(self.tok, ctx, self.gen, save_dir=self.root, clock=self.clock, archive=self.arc, memory_tokens=200, **kw)

    def test_every_exchange_is_filed_with_a_session_id(self):
        s = self.session()
        s.handle("hello")
        s.handle("how are you")
        items = A.ChatArchive(self.path).items
        self.assertEqual([i["user"] for i in items], ["hello", "how are you"])
        self.assertEqual(len({i["session"] for i in items}), 1)

    def test_reset_starts_a_new_chat_in_the_archive_and_keeps_the_old_one(self):
        s = self.session()
        s.handle("hello")
        s.session_id = "s-first"
        s._exchanges = []
        s.handle("/reset")
        self.assertNotEqual(s.session_id, "s-first")
        s.handle("hello again")
        self.assertEqual(len(A.ChatArchive(self.path)), 2)

    def test_the_ai_is_shown_an_exchange_from_200_chats_ago(self):
        old = A.ChatArchive(self.path, clock=self.clock)
        old.append("s0001", "maria loves mango", "that is sweet", t=T0 - 400 * DAY)
        for i in range(2, 201):                                                  # 199 more chats about other things
            old.append("s%04d" % i, "hello how are you today", "fine thanks", t=T0 - (400 - i) * DAY)
        s = self.session()
        self.assertEqual(s.archive.sessions(), 200)
        out, _ = s.handle("do you remember maria and the mango")
        self.assertTrue(s.last_archive_recalled)
        self.assertEqual(s.last_archive_recalled[0], 1)                           # the OLDEST chat was found
        self.assertIn("old chats recalled: #1", out)
        ids = self.gen.calls[0]
        self.assertIn(MEMORY, ids)
        text = self.tok.decode([i for i in ids[ids.index(MEMORY) + 1:] if i != 0])
        self.assertIn("earlier chat", text)
        self.assertIn("mango", text)

    def test_exchanges_still_inside_the_window_are_not_repeated_in_the_memory_block(self):
        s = self.session()
        s.handle("i adopted a puppy named Rex")
        out, _ = s.handle("tell me about the puppy")
        self.assertEqual(s.last_archive_recalled, [])                              # the puppy exchange is still on screen for the model

    def test_nothing_recalled_when_no_word_matches(self):
        s = self.session()
        s.handle("hello")
        s.handle("quantum chromodynamics")
        self.assertEqual(s.last_archive_recalled, [])

    def test_the_memory_block_never_eats_the_conversation(self):
        old = A.ChatArchive(self.path, clock=self.clock)
        for i in range(60):
            old.append("s%d" % i, "mango " * 60, "mango " * 60, t=T0 - i * DAY)
        s = self.session(ctx=256)
        out, _ = s.handle("mango")
        self.assertLessEqual(len(self.gen.calls[0]), 256 - s.settings["max_new"] + 1 + 1)
        self.assertNotIn("could not", out)

    def test_recall_command(self):
        s = self.session()
        self.assertIn("empty", s.handle("/recall")[0])
        s.handle("maria loves mango")
        s.handle("/reset")
        text, _ = s.handle("/recall mango")
        self.assertIn("#1", text)
        self.assertIn("mango", text)
        self.assertIn("No old chat", s.handle("/recall zebra")[0])

    def test_recall_command_explains_itself_when_the_archive_is_off(self):
        s = chatrepl.ChatSession(self.tok, 512, FakeGen(self.tok), save_dir=self.root, clock=self.clock)
        self.assertIn("--archive", s.handle("/recall mango")[0])
        s.handle("hello")
        self.assertEqual(os.listdir(self.root), [])

    def test_a_broken_archive_file_does_not_stop_the_chat(self):
        s = self.session()
        os.makedirs(self.path + "x")
        s.archive.path = self.path + "x"                                           # a folder where the file should be
        out, _ = s.handle("hello")
        self.assertIn("ai> sure thing", out)
        self.assertIn("could not save to the chat archive", out)


if __name__ == "__main__":
    unittest.main()
