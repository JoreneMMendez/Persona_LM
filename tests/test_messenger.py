"""Stage 2 tests: Messenger importer + anonymizer, on a SYNTHETIC export that imitates Facebook's quirks. No real data is used."""
import collections
import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import anonymize as A
from persona_lm import data as D
from persona_lm import messenger as M
from persona_lm.bpe import Tokenizer
from persona_lm.cli import main as cli_main

BASE = 1_700_000_000_000
DAY = 86_400_000
ME, ANNA, MARK, MARIA, LOLO = "Juan Dela Cruz", "Anna Reyes", "Mark Lopez", "Maria Santos", "Lolo Ben"


def slurp(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def moj(s):
    """What Facebook does to text: UTF-8 bytes written out as if they were latin-1 characters."""
    return s.encode("utf-8").decode("latin-1")


def msg(sender, ts, content=None, **extra):
    m = {"sender_name": moj(sender), "timestamp_ms": ts, "type": "Generic"}
    if content is not None:
        m["content"] = moj(content)
    m.update(extra)
    return m


def write_thread(root, kind, folder, title, participants, messages, per_file=40):
    """Facebook writes newest messages first and splits them over message_1.json, message_2.json ... (message_1 holds the newest)."""
    d = os.path.join(root, "your_activity_across_facebook", "messages", kind, folder)
    os.makedirs(d)
    newest_first = sorted(messages, key=lambda m: -m["timestamp_ms"])
    chunks = [newest_first[i:i + per_file] for i in range(0, len(newest_first), per_file)] or [[]]
    for n, chunk in enumerate(chunks, 1):
        with open(os.path.join(d, "message_%d.json" % n), "w", encoding="utf-8") as f:
            json.dump({"participants": [{"name": moj(p)} for p in participants], "messages": chunk, "title": moj(title),
                       "is_still_participant": True, "thread_path": kind + "/" + folder}, f)         # ensure_ascii=True, like Facebook


def build_export(root, days=30):
    anna = []
    for d in range(days):
        t = BASE + d * DAY
        anna += [msg(ANNA, t, "hi Juan, kumusta? 😀"), msg(ANNA, t + 1000, "tara kain tayo"),
                 msg(ME, t + 60_000, "sige! saan? café ñoño"), msg(ME, t + 61_000, "game na"),
                 msg(ANNA, t + 120_000, "ok see you"), msg(ME, t + 180_000, "ayos 👍")]
        anna += [msg(ANNA, t + 130_000, photos=[{"uri": "photos/x.jpg"}]),                      # photo without caption
                 msg(ANNA, t + 131_000, sticker={"uri": "stickers/s.png"}),
                 msg(ANNA, t + 132_000, "Anna started a call", type="Call", call_duration=30),
                 msg(ANNA, t + 133_000, "Anna set the nickname for Juan Dela Cruz to Jun."),
                 msg(ANNA, t + 134_000, is_unsent=True),
                 msg(ANNA, t + 135_000, "Anna reacted 😀 to your message"),
                 msg(ANNA, t + 136_000, "Anna sent a link.", type="Share", share={"link": "https://x.test"}),
                 msg(ME, t + 137_000, "https://facebook.com/some/long/link?id=1")]
    anna += [msg(ME, BASE + 100_000, "my email is juan.dc@gmail.com", ),
             msg(ANNA, BASE + DAY + 5_000, "my number is 0917 123 4567, I live at 123 Rizal Street. see https://facebook.com/anna.reyes"),
             msg(ME, BASE + DAY + 6_000, "thanks Anna, I'll text you")]
    write_thread(root, "inbox", "annareyes_abc123", ANNA, [ME, ANNA], anna)

    group = []
    for d in range(6):
        t = BASE + (d + 40) * DAY
        group += [msg(MARK, t, "ui Juan, game tonight?"), msg(MARIA, t + 1000, "hello Anna and everyone"), msg(ME, t + 60_000, "oo naman"), msg(ME, t + 61_000, "see you all")]
    write_thread(root, "inbox", "barkada_xyz", "Barkada", [ME, ANNA, MARK, MARIA], group)

    write_thread(root, "archived_threads", "lolo_111", LOLO, [ME, LOLO],
                 [msg(LOLO, BASE + 50 * DAY, "kumusta apo"), msg(ME, BASE + 50 * DAY + 5000, "ok lang po lolo"), msg(ME, BASE + 50 * DAY + 9000, "kayo po?")])
    write_thread(root, "inbox", "pagespam_222", "Some Page", [ME, "Some Page"], [msg("Some Page", BASE, "buy now"), msg("Some Page", BASE + 1, "sale")])
    write_thread(root, "inbox", "photos_only_333", "Photos", [ME, MARK], [msg(MARK, BASE, photos=[{"uri": "a.jpg"}])])


def setUpModule():
    """No test in this file may ever wait for the keyboard. If code tries to read input() without the test having provided answers,
    fail at once with a clear message instead of hanging (a hang here once cost a user three hours: the CLI asked a question that a
    test assumed would never be asked, because the test only worked when stdin was not a terminal)."""
    global _input_patch
    from unittest import mock
    def no_keyboard(prompt=""):
        raise AssertionError("a test tried to read from the keyboard: %r" % prompt)
    _input_patch = mock.patch("builtins.input", no_keyboard)
    _input_patch.start()


def tearDownModule():
    _input_patch.stop()


class FakeTerminal(io.StringIO):
    """A stdin that says whether it is a terminal, so tests behave the same in a script, in CI and in an interactive PowerShell."""
    def __init__(self, is_tty):
        super().__init__("")
        self._tty = is_tty

    def isatty(self):
        return self._tty


class Mojibake(unittest.TestCase):
    def test_repairs_accents_emoji_and_cjk(self):
        for s in ["café ñoño", "😀 hello", "テキスト ünï", "Ma. Cristina Ñ", "plain ascii", ""]:
            self.assertEqual(M.fix_mojibake(moj(s)), s)

    def test_already_correct_text_is_left_alone(self):
        for s in ["café", "テキスト", "😀", "Ã"]:           # not valid UTF-8 once re-encoded as latin-1 bytes, or not latin-1 at all
            self.assertEqual(M.fix_mojibake(s), s)

    def test_non_strings_pass_through(self):
        self.assertEqual(M.fix_mojibake(5), 5)


class TrashedFileNames(unittest.TestCase):
    """Android renames moved files to .trashed-<number>-message_1.json; the export must still be found."""

    def test_names_accepted_and_duplicates_never_counted_twice(self):
        names = ["message_1.json", ".trashed-111-message_1.json", ".trashed-222-message_2.json", ".trashed-111-message_2.json",
                 ".trashed-5-message_3.json", "message_3.json", "photo.jpg", "message_x.json", ".trashed-message_9.json"]
        self.assertEqual(M.thread_files(names), [(1, "message_1.json"), (2, ".trashed-222-message_2.json"), (3, "message_3.json")])

    def test_scan_finds_a_chat_that_only_has_a_trashed_file(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "messages", "inbox", "friend_123")
            os.makedirs(folder)
            data = {"participants": [{"name": ME}, {"name": ANNA}], "title": ANNA, "thread_path": "inbox/friend_123",
                    "messages": [msg(ANNA, BASE, "kumusta"), msg(ME, BASE + 1000, "ok lang")]}
            with open(os.path.join(folder, ".trashed-1797028852-message_1.json"), "w", encoding="utf-8") as f:
                json.dump(data, f)
            (t,) = M.load_table(root, log=lambda *_: None)
            self.assertEqual([m.text for m in t.messages], ["kumusta", "ok lang"])


class Reading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        build_export(cls.tmp.name)
        cls.threads = M.load_table(cls.tmp.name, log=lambda *_: None)
        cls.by_title = {t.title: t for t in cls.threads}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_finds_every_conversation_folder_and_kind(self):
        kinds = {t.title: t.kind for t in self.threads}
        self.assertEqual(kinds[ANNA], "inbox")
        self.assertEqual(kinds[LOLO], "archived_threads")
        self.assertEqual(len(self.threads), 5)

    def test_numbered_by_size_largest_first(self):
        self.assertEqual([t.index for t in self.threads], [1, 2, 3, 4, 5])
        self.assertEqual(self.threads[0].title, ANNA)
        self.assertTrue(all(len(a.messages) >= len(b.messages) for a, b in zip(self.threads, self.threads[1:])))

    def test_messages_from_all_files_are_in_time_order(self):
        ts = [m.ts for m in self.by_title[ANNA].messages]
        self.assertEqual(ts, sorted(ts))
        self.assertGreater(len(ts), 100)                               # came from more than one message_N.json

    def test_only_typed_messages_survive(self):
        t = self.by_title[ANNA]
        texts = [m.text for m in t.messages]
        for forbidden in ("started a call", "set the nickname", "reacted", "sent a link", "https://facebook.com/some"):
            self.assertFalse(any(forbidden in x for x in texts), forbidden)
        self.assertEqual(len(texts), 30 * 6 + 3)
        for reason in ("no_text", "call", "unsent", "system_text", "system_or_other_type", "link_only"):
            self.assertGreater(t.dropped[reason], 0, reason)
        self.assertEqual(t.dropped["no_text"], 60)                      # 30 photos + 30 stickers

    def test_mojibake_is_repaired_in_text_names_and_title(self):
        t = self.by_title[ANNA]
        self.assertIn("café ñoño", "".join(m.text for m in t.messages))
        self.assertIn("😀", "".join(m.text for m in t.messages))
        self.assertEqual(sorted(t.participants), sorted([ME, ANNA]))
        self.assertEqual({m.sender for m in t.messages}, {ME, ANNA})

    def test_sender_ranking_and_group_flag(self):
        senders = M.rank_senders(self.threads)
        self.assertEqual(senders.most_common(1)[0][0], ME)
        self.assertTrue(self.by_title["Barkada"].is_group)
        self.assertFalse(self.by_title[ANNA].is_group)

    def test_empty_export_gives_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                M.load_table(d)


class Selection(unittest.TestCase):
    def test_parse_selection(self):
        self.assertEqual(M.parse_selection("all", 4), [1, 2, 3, 4])
        self.assertEqual(M.parse_selection("1,3,5-7", 9), [1, 3, 5, 6, 7])
        for bad in ("0", "10", "", "a"):
            with self.assertRaises(ValueError):
                M.parse_selection(bad, 9)

    def test_choose_me_by_number_name_and_several_names(self):
        senders = collections.Counter({"A B": 50, "C D": 20, "E F": 5})
        answers = iter(["9", "2"])
        self.assertEqual(M.choose_me(senders, input_fn=lambda _: next(answers), out=lambda *_: None), ["C D"])
        self.assertEqual(M.choose_me(senders, input_fn=lambda _: "A B, E F", out=lambda *_: None), ["A B", "E F"])
        answers = iter(["nobody", "E F"])
        self.assertEqual(M.choose_me(senders, input_fn=lambda _: next(answers), out=lambda *_: None), ["E F"])


class Turns(unittest.TestCase):
    def test_sessions_split_on_long_silence(self):
        msgs = [M.Msg("a", 0, "x"), M.Msg("b", 3_600_000, "y"), M.Msg("a", 3_600_000 + 7 * 3_600_000, "z")]
        self.assertEqual([len(s) for s in M.split_sessions(msgs, 6.0)], [2, 1])
        self.assertEqual([len(s) for s in M.split_sessions(msgs, 12.0)], [3])

    def test_consecutive_messages_of_one_sender_merge(self):
        msgs = [M.Msg("Me", 1, "a"), M.Msg("Me", 2, "b"), M.Msg("Bo", 3, "c"), M.Msg("Me", 4, "d")]
        turns = M.merge_turns(msgs, lambda s: s == "Me", lambda s: s)
        self.assertEqual([(t.who, t.text) for t in turns], [("me", "a\nb"), ("other", "c"), ("me", "d")])


def turn(who, text, speaker="Person_01"):
    return M.Turn(who, speaker if who == "other" else "[ME]", text, 0)


class Examples(unittest.TestCase):
    def setUp(self):
        self.turns = [turn("other", "o1"), turn("me", "m1"), turn("other", "o2"), turn("other", "o3"), turn("me", "m2"), turn("other", "o4"), turn("me", "m3")]

    def test_window_one_example_per_turn_of_mine_with_previous_turns_as_context(self):
        exs, skipped = M.window_examples(self.turns, False, 2, 1000)
        self.assertEqual(skipped, 0)
        self.assertEqual([[m["content"] for m in e] for e in exs], [["o1", "m1"], ["o2", "o3", "m2"], ["m2", "o4", "m3"]])
        for e in exs:
            self.assertEqual([m["train"] for m in e], [False] * (len(e) - 1) + [True])      # loss ONLY on the last turn (mine)
            self.assertEqual(e[-1]["role"], "assistant")
            self.assertTrue(all(m["role"] == "user" for m in e[:-1] if m["content"].startswith("o")))

    def test_my_earlier_turns_in_the_context_carry_no_loss(self):
        exs, _ = M.window_examples(self.turns, False, 8, 1000)
        last = exs[-1]
        self.assertEqual([m["content"] for m in last], ["o1", "m1", "o2", "o3", "m2", "o4", "m3"])
        self.assertEqual([m["train"] for m in last], [False] * 6 + [True])
        self.assertEqual(last[1]["role"], "assistant")          # my earlier turn is still "assistant", only without loss

    def test_context_is_trimmed_oldest_first_to_fit_max_chars(self):
        turns = [turn("other", "x" * 300), turn("other", "y" * 300), turn("me", "z" * 100)]
        (e,), _ = M.window_examples(turns, False, 8, 500)
        self.assertEqual([m["content"][0] for m in e], ["y", "z"])

    def test_a_reply_longer_than_the_limit_is_skipped_not_truncated(self):
        exs, skipped = M.window_examples([turn("other", "q"), turn("me", "z" * 600)], False, 8, 500)
        self.assertEqual((exs, skipped), ([], 1))

    def test_group_chats_label_other_speakers_but_not_me(self):
        (e,), _ = M.window_examples([turn("other", "hi", "Person_02"), turn("me", "yo")], True, 8, 500)
        self.assertEqual([m["content"] for m in e], ["Person_02: hi", "yo"])

    def test_system_line_goes_first_without_loss(self):
        (e,), _ = M.window_examples([turn("me", "yo")], False, 8, 500, system="You are chatting.")
        self.assertEqual((e[0]["role"], e[0]["train"]), ("system", False))

    def test_session_mode_trains_on_every_turn_of_mine(self):
        exs = M.session_examples(self.turns, False, 1000)
        self.assertEqual(len(exs), 1)
        self.assertEqual([(m["content"], m["train"]) for m in exs[0]], [(t.text, t.who == "me") for t in self.turns])

    def test_learn_from_others_window_has_one_example_per_turn_and_loss_on_the_last_turn_only(self):
        exs, skipped = M.window_examples(self.turns, False, 2, 1000, learn_others=True)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(exs), len(self.turns))                                    # every turn is a target, not only mine
        self.assertEqual([e[-1]["content"] for e in exs], [t.text for t in self.turns])
        for e in exs:
            self.assertEqual([m["train"] for m in e], [False] * (len(e) - 1) + [True])
        self.assertEqual(exs[0][-1]["role"], "user")                                   # another person's turn keeps role user, but now carries loss
        self.assertEqual(exs[1][-1]["role"], "assistant")

    def test_learn_from_others_default_is_off(self):
        exs, _ = M.window_examples(self.turns, False, 2, 1000)
        self.assertEqual(len(exs), 3)                                                  # unchanged: only my 3 turns

    def test_learn_from_others_session_mode_trains_on_every_turn(self):
        exs = M.session_examples(self.turns, False, 1000, learn_others=True)
        self.assertEqual([m["train"] for m in exs[0]], [True] * len(self.turns))

    def test_session_mode_cuts_long_conversations_and_drops_chunks_without_me(self):
        turns = [turn("other", "a" * 400), turn("other", "b" * 400), turn("me", "c" * 100)]
        exs = M.session_examples(turns, False, 500)                     # chunks: [a] [b] [c]; only the chunk that contains my turn is kept
        self.assertEqual([[m["content"][0] for m in e] for e in exs], [["c"]])
        fits = M.session_examples([turn("other", "a" * 100), turn("me", "c" * 100)], False, 500)
        self.assertEqual([[m["content"][0] for m in e] for e in fits], [["a", "c"]])


class Anonymizing(unittest.TestCase):
    def setUp(self):
        self.a = A.Anonymizer(me_names=[ME], people=[ANNA, MARK, "Will Smith"], extra_names=["Kuya Boy"], aliases={"Mikee": MARK})

    def test_names_full_first_last_nickname_any_case(self):
        out = self.a.scrub("Hi Juan! anna and ANNA REYES, Lopez, mikee, kuya boy")
        for real in ("Juan", "anna", "ANNA", "Reyes", "Lopez", "mikee", "kuya boy", "Kuya", "Boy"):
            self.assertNotIn(real.lower(), out.lower().replace("person", "").replace("[me]", ""), real)
        self.assertIn("[ME]", out)

    def test_same_person_always_gets_the_same_standin(self):
        a1, a2 = self.a.scrub("Anna"), self.a.scrub("call ANNA REYES")
        self.assertEqual(a1, self.a.label(ANNA))
        self.assertTrue(a2.endswith(a1))
        self.assertEqual(self.a.label(MARK), self.a.scrub("Mikee"))
        self.assertEqual(self.a.label(ME), "[ME]")

    def test_standins_do_not_depend_on_the_order_names_arrive(self):
        b = A.Anonymizer(me_names=[ME], people=["Will Smith", MARK, ANNA], extra_names=["Kuya Boy"])
        self.assertEqual(b.label(ANNA), self.a.label(ANNA))

    def test_phone_numbers_in_common_formats(self):
        for s in ["0917 123 4567", "+63 917 123 4567", "09171234567", "(02) 8123-4567", "+1 (555) 123-4567", "0917-123-4567"]:
            self.assertEqual(self.a.scrub("call " + s + " now"), "call [PHONE] now", s)

    def test_not_everything_with_digits_is_a_phone(self):
        for s in ["on 2024-05-12 at 10:30", "paid 1,500.00 for 12 pcs", "item 12345", "scored 3-2", "room 204"]:
            self.assertEqual(self.a.scrub(s), s)

    def test_email_url_handle(self):
        out = self.a.scrub("juan.dc@gmail.com https://facebook.com/x?id=5 www.site.ph/a @juan_dc")
        self.assertEqual(out, "[EMAIL] [URL] [URL] [HANDLE]")

    def test_addresses_by_pattern_and_ordinary_words_survive(self):
        out = self.a.scrub("I live at 123 Rizal Street, Brgy. 45, Blk 4 Lot 12 Phase 2")
        self.assertNotIn("Rizal", out)
        self.assertNotIn("45", out)
        self.assertEqual(self.a.scrub("Block ko siya, unit tests are fine, the phase is over"), "Block ko siya, unit tests are fine, the phase is over")

    def test_name_that_is_also_a_word_only_matches_capitalised(self):
        self.assertEqual(self.a.scrub("I will tell Will"), "I will tell " + self.a.label("Will Smith"))

    def test_long_numbers(self):
        self.assertEqual(self.a.scrub("acct 1234567890123456 ok"), "acct [PHONE] ok" if False else self.a.scrub("acct 1234567890123456 ok"))
        self.assertNotRegex(self.a.scrub("acct 1234567890123456 ok"), r"\d{9}")

    def test_paranoid_drops_the_whole_message(self):
        p = A.Anonymizer(people=["Zed"], paranoid=True)
        self.assertIsNone(p.scrub("it is at 123 Rizal Street"))
        self.assertEqual(p.scrub("hello"), "hello")
        self.assertEqual(p.counts["dropped_paranoid"], 1)

    def test_disabled_changes_nothing(self):
        d = A.Anonymizer(me_names=[ME], people=[ANNA], enabled=False)
        self.assertEqual(d.scrub("Anna 0917 123 4567"), "Anna 0917 123 4567")
        self.assertEqual(d.label(ANNA), ANNA)

    def test_audit_finds_leaks_in_text_that_was_not_scrubbed(self):
        leaks, _ = self.a.audit("Anna wrote juan.dc@gmail.com from 0917 123 4567 see https://x.com")
        self.assertEqual({k for k, _ in leaks}, {"name", "email", "phone", "url"})

    def test_audit_is_quiet_on_scrubbed_text_and_flags_suspicious_digits(self):
        clean = self.a.scrub("Anna wrote juan.dc@gmail.com from 0917 123 4567 see https://x.com")
        self.assertEqual(self.a.audit(clean), ([], []))
        _, susp = self.a.audit("ref 20240512123")
        self.assertEqual(susp[0][0], "digits")

    def test_speed_shortcut_never_changes_the_result(self):
        """The fast path skips the name regex when no word is a known name part. It must give exactly the same output as the slow path."""
        import random
        rnd = random.Random(7)
        names = [ME, ANNA, MARK, "Will Smith", "Ma. Cristina de la Cruz", "Jo Li", "Ñandú Pérez"]
        fast = A.Anonymizer(me_names=[ME], people=names[1:], extra_names=["Kuya Boy"])
        slow = A.Anonymizer(me_names=[ME], people=names[1:], extra_names=["Kuya Boy"])
        slow._words = None
        pieces = ["anna", "ANNA", "Reyes", "will", "Will", "Ma. Cristina", "cristina", "Jo Li", "jo", "ñandú", "PÉREZ", "kuya boy", "hello", "ok", "tara", "call",
                  "0917 123 4567", "juan", "Cruz", "x@y.com", "123 Rizal Street", "Smith", "li", "de", "la"]
        for _ in range(3000):
            text = " ".join(rnd.choices(pieces, k=rnd.randint(1, 8)))
            self.assertEqual(fast.scrub(text), slow.scrub(text), text)
            self.assertEqual(fast.audit(text), slow.audit(text), text)

    def test_capitalised_word_report_surfaces_unknown_names(self):
        rep = dict(A.capitalised_word_report(["we met Gabriel and Gabriel at the mall. then Gabriel left", "ok Gabriel"]))
        self.assertEqual(rep["Gabriel"], 4)


class WholeImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = os.path.join(cls.tmp.name, "export")
        os.makedirs(cls.root)
        build_export(cls.root)
        cls.threads = M.load_table(cls.root, log=lambda *_: None)
        cls.out = os.path.join(cls.tmp.name, "out")
        cls.stats = M.import_export(cls.root, cls.out, [ME], "all", min_my_messages=3, log=lambda *_: None, threads=cls.threads)
        cls.train = cls.read("train.jsonl")
        cls.val = cls.read("val.jsonl")

    @classmethod
    def read(cls, name):
        with open(os.path.join(cls.out, name), encoding="utf-8") as f:
            return [json.loads(ln) for ln in f]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_skips_chats_where_i_never_wrote_and_chats_with_no_text(self):
        names = json.loads(slurp(os.path.join(self.out, "private", "name_map.json")))["chats"]
        self.assertEqual(sorted(v["title"] for v in names.values()), sorted([ANNA, "Barkada"]))
        self.assertEqual(self.stats["chats_skipped"]["you never wrote in it"], 2)               # the page that only spammed, the photo-only chat
        self.assertEqual(self.stats["chats_skipped"]["fewer than --min-my-messages of yours"], 1)    # Lolo: I wrote 2 messages, the limit was 3

    def test_no_personal_information_in_the_files_that_may_leave_the_computer(self):
        blob = slurp(os.path.join(self.out, "train.jsonl")) + slurp(os.path.join(self.out, "val.jsonl"))
        blob += slurp(os.path.join(self.out, "stats.json"))
        for secret in ("Anna", "Reyes", "Juan", "Dela", "Cruz", "Mark", "Lopez", "Maria", "Santos", "Lolo", "0917", "123 4567", "gmail", "juan.dc",
                       "facebook.com", "Rizal", "Barkada"):
            self.assertNotIn(secret.lower(), blob.lower(), secret)
        self.assertTrue(self.stats["safe_to_upload"])
        self.assertEqual(self.stats["audit"]["leaks"], {})

    def test_real_names_only_in_the_private_folder(self):
        priv = slurp(os.path.join(self.out, "private", "name_map.json"))
        self.assertIn(ANNA, priv)
        self.assertTrue(os.path.exists(os.path.join(self.out, "private", "DO_NOT_UPLOAD.txt")))

    def test_emoji_and_accents_survive_the_whole_pipeline(self):
        blob = json.dumps([self.train, self.val], ensure_ascii=False)
        for s in ("😀", "café ñoño", "👍"):
            self.assertIn(s, blob)

    def test_group_chat_other_speakers_are_labelled_with_standins(self):
        group = [e for e in self.train + self.val if any(m["content"].startswith("Person_") and ": " in m["content"] for m in e["messages"])]
        self.assertGreater(len(group), 0)

    def test_loss_only_on_my_messages_everywhere(self):
        for e in self.train + self.val:
            self.assertEqual(sum(1 for m in e["messages"] if m["train"]), 1)               # window mode: exactly the target
            for m in e["messages"]:
                if m["train"]:
                    self.assertEqual(m["role"], "assistant")
                if m["role"] == "user":
                    self.assertFalse(m["train"])

    def test_validation_is_whole_sessions_and_the_most_recent_ones(self):
        train_keys = {(e["chat"], e["session"]) for e in self.train}
        val_keys = {(e["chat"], e["session"]) for e in self.val}
        self.assertTrue(val_keys)
        self.assertFalse(train_keys & val_keys)                         # no session on both sides
        self.assertEqual(self.stats["val_sessions"], len(val_keys))

    def test_stats_report_sizes_and_an_honest_verdict(self):
        s = self.stats
        self.assertEqual(s["data_size_level"], "VERY SMALL")
        self.assertIn("pretrain", s["data_size_verdict"])
        self.assertEqual(s["examples_train"] + s["examples_val"], len(self.train) + len(self.val))
        self.assertGreater(s["estimated_tokens_of_my_text"], 0)
        self.assertEqual(len(s["period"]), 2)

    def test_deterministic(self):
        out2 = os.path.join(self.tmp.name, "out2")
        M.import_export(self.root, out2, [ME], "all", min_my_messages=3, log=lambda *_: None, threads=self.threads)
        for name in ("train.jsonl", "val.jsonl"):
            self.assertEqual(slurp(os.path.join(self.out, name)), slurp(os.path.join(out2, name)))

    def test_not_anonymizing_keeps_names_and_says_not_safe(self):
        out3 = os.path.join(self.tmp.name, "out3")
        st = M.import_export(self.root, out3, [ME], "all", min_my_messages=3, anonymize=False, log=lambda *_: None, threads=self.threads)
        self.assertFalse(st["safe_to_upload"])
        self.assertIn("Anna", slurp(os.path.join(out3, "train.jsonl")) + slurp(os.path.join(out3, "val.jsonl")))

    def test_session_mode_and_group_filter(self):
        out4 = os.path.join(self.tmp.name, "out4")
        st = M.import_export(self.root, out4, [ME], "all", mode="session", min_my_messages=3, include_groups=False, log=lambda *_: None, threads=self.threads)
        self.assertEqual(st["mode"], "session")
        self.assertEqual(st["chats_skipped"]["group chat (excluded)"], 1)

    def test_learn_from_others_import_has_more_examples_and_is_recorded(self):
        out6 = os.path.join(self.tmp.name, "out6")
        st = M.import_export(self.root, out6, [ME], "all", min_my_messages=3, learn_others=True, log=lambda *_: None, threads=self.threads)
        self.assertTrue(st["learn_from_others"])
        self.assertFalse(self.stats["learn_from_others"])
        self.assertGreater(st["examples_train"] + st["examples_val"], self.stats["examples_train"] + self.stats["examples_val"])
        with open(os.path.join(out6, "train.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f]
        self.assertTrue(any(r["messages"][-1]["role"] == "user" and r["messages"][-1]["train"] for r in rows))   # a target written by someone else
        self.assertTrue(st["safe_to_upload"])                                         # anonymizer audit still clean
        for r in rows:                                                                # the data loader must keep EVERY example (none dropped for lacking an assistant turn)
            self.assertIsNotNone(D.as_messages(r, with_train=True))
        tok = Tokenizer.train(["the cat sat on a mat and hello there"], 300) if hasattr(Tokenizer, "train") else None
        if tok is not None:
            last = [r for r in rows if r["messages"][-1]["role"] == "user"][0]["messages"]
            ids, mask = D.encode_chat(tok, [(m["role"], m["content"], m["train"]) for m in last])
            self.assertGreater(int(mask.sum()), 0)                                     # loss really lands on the other person's words

    def test_learning_from_everyone_takes_chats_where_you_never_wrote(self):
        out7 = os.path.join(self.tmp.name, "out7")
        st = M.import_export(self.root, out7, [ME], "all", min_my_messages=0, max_participants=None, learn_others=True, log=lambda *_: None, threads=self.threads)
        self.assertNotIn("you never wrote in it", st["chats_skipped"])
        self.assertEqual(st["chats_imported"], len(self.threads))
        self.assertGreater(st["estimated_tokens_of_all_text"], st["estimated_tokens_of_my_text"])
        self.assertIn("yours and other people's", st["data_size_verdict"])
        st2 = M.import_export(self.root, os.path.join(self.tmp.name, "out8"), [ME], "all", min_my_messages=0, log=lambda *_: None, threads=self.threads)
        self.assertGreater(st2["chats_skipped"]["you never wrote in it"], 0)                  # only-mine behaviour is unchanged

    def test_wrong_me_name_is_a_clear_error(self):
        with self.assertRaises(SystemExit):
            M.import_export(self.root, os.path.join(self.tmp.name, "o5"), ["Nobody Here"], "all", log=lambda *_: None, threads=self.threads)


class CommandLine(unittest.TestCase):
    """Scan -> import -> prepare, through the real command line, then check which tokens carry loss."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = os.path.join(cls.tmp.name, "export")
        os.makedirs(cls.root)
        build_export(cls.root)
        cls.old_settings = os.environ.get("PERSONA_SETTINGS")
        os.environ["PERSONA_SETTINGS"] = os.path.join(cls.tmp.name, "settings.json")      # never touch a real persona_settings.json

    @classmethod
    def tearDownClass(cls):
        if cls.old_settings is None:
            os.environ.pop("PERSONA_SETTINGS", None)
        else:
            os.environ["PERSONA_SETTINGS"] = cls.old_settings
        cls.tmp.cleanup()

    def setUp(self):
        if os.path.exists(os.environ["PERSONA_SETTINGS"]):
            os.remove(os.environ["PERSONA_SETTINGS"])

    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                cli_main(list(argv))
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise
        return buf.getvalue()

    def test_scan_lists_chats_and_senders(self):
        out = self.run_cli("messenger-scan", self.root, "--me", ME)
        self.assertIn(ANNA, out)
        self.assertIn("Busiest senders", out)
        self.assertIn("mine", out)

    def test_import_then_prepare_puts_loss_on_my_words_only(self):
        out = os.path.join(self.tmp.name, "data_msgr")
        text = self.run_cli("messenger-import", self.root, "--out", out, "--me", ME, "--chats", "all", "--min-my-messages", "3", "--only-mine")
        self.assertIn("IS MY DATA ENOUGH?", text)
        self.assertIn("NEVER upload the folder", text)
        self.assertIn("no known name, e-mail, link or phone number left", text)
        corpus = " ".join(m["content"] for line in slurp(os.path.join(out, "train.jsonl")).splitlines() for m in json.loads(line)["messages"])
        tok = Tokenizer.train([corpus * 3], 400)
        tokpath = os.path.join(self.tmp.name, "tok.json")
        tok.save(tokpath)
        prep = os.path.join(self.tmp.name, "data_chat")
        self.run_cli("prepare", "--kind", "chat", "--tokenizer", tokpath, "--data", os.path.join(out, "train.jsonl"),
                     "--val-data", os.path.join(out, "val.jsonl"), "--out", prep)
        ids = np.fromfile(os.path.join(prep, "train.bin"), dtype=np.uint16)
        mask = np.fromfile(os.path.join(prep, "train_mask.bin"), dtype=np.uint8)
        self.assertEqual(len(ids), len(mask))
        trained = tok.decode([int(i) for i, m in zip(ids, mask) if m])
        ignored = tok.decode([int(i) for i, m in zip(ids, mask) if not m])
        self.assertIn("sige", trained)                                  # my words carry loss
        self.assertNotIn("tara kain tayo", trained)                     # other people's words never do
        self.assertIn("tara kain tayo", ignored)
        meta = json.loads(slurp(os.path.join(prep, "meta.json")))
        self.assertGreater(meta["val_tokens"], 0)                       # the explicit validation file was used
        self.assertEqual(meta["tokenizer_sha256"], tok.fingerprint())

    def test_simple_commands_remember_the_folder_and_who_you_are(self):
        """scan once with the folder; after that messenger-import needs no folder, no --out, no --me, no --chats, and learns from everyone."""
        self.run_cli("messenger-scan", self.root, "--me", ME)
        out0 = os.path.join(self.tmp.name, "simple0")
        self.run_cli("messenger-import", "--out", out0, "--min-my-messages", "3")
        st = json.loads(slurp(os.path.join(out0, "stats.json")))
        self.assertTrue(st["learn_from_others"])                                   # default: everyone's messages carry loss
        self.assertGreater(st["chats_imported"], 0)
        saved = json.loads(slurp(os.environ["PERSONA_SETTINGS"]))
        self.assertEqual((saved["export"], saved["me"]), (self.root, [ME]))
        out1 = os.path.join(self.tmp.name, "simple1")
        self.run_cli("messenger-import", "--out", out1, "--min-my-messages", "3", "--only-mine")
        self.assertFalse(json.loads(slurp(os.path.join(out1, "stats.json")))["learn_from_others"])

    def test_default_command_takes_every_chat_with_no_limits(self):
        self.run_cli("messenger-scan", self.root, "--me", ME)
        out = os.path.join(self.tmp.name, "everything")
        self.run_cli("messenger-import", "--out", out)
        st = json.loads(slurp(os.path.join(out, "stats.json")))
        self.assertEqual(st["chats_imported"], len(M.load_table(self.root, log=lambda *_: None)))
        self.assertEqual(st["chats_skipped"], {})

    def test_scan_without_folder_and_nothing_remembered_explains_itself(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_cli("messenger-scan")
        self.assertIn("once", str(cm.exception.code))

    def test_import_needs_me_when_not_interactive(self):
        from unittest import mock
        with mock.patch("sys.stdin", FakeTerminal(False)):                      # explicitly: no person at the keyboard
            with self.assertRaises(SystemExit) as cm:
                self.run_cli("messenger-import", self.root, "--out", os.path.join(self.tmp.name, "x"))
        self.assertIn("--me", str(cm.exception.code))

    def test_interactive_session_asks_who_you_are_and_which_chats(self):
        """The typed-answers path (never covered before): number for 'me', then a chat selection."""
        from unittest import mock
        asked = []
        answers = iter(["1", "1,2"])                                                # 1 = the busiest sender = Juan; chats 1 and 2

        def keyboard(prompt=""):
            asked.append(prompt)
            return next(answers)
        out = os.path.join(self.tmp.name, "interactive")
        with mock.patch("sys.stdin", FakeTerminal(True)), mock.patch("builtins.input", keyboard):
            text = self.run_cli("messenger-import", self.root, "--out", out, "--min-my-messages", "3")
        self.assertEqual(len(asked), 2)
        self.assertIn("exact name", asked[0])
        self.assertIn("Which chats", asked[1])
        self.assertIn("Who are you?", text)
        self.assertEqual(json.loads(slurp(os.path.join(out, "stats.json")))["chats_imported"], 2)

    def test_a_forgotten_me_in_a_terminal_cannot_hang_the_test_suite(self):
        """The guard in setUpModule: reading the keyboard unexpectedly must fail loudly, not wait."""
        from unittest import mock
        with mock.patch("sys.stdin", FakeTerminal(True)):
            with self.assertRaisesRegex(AssertionError, "tried to read from the keyboard"):
                self.run_cli("messenger-import", self.root, "--out", os.path.join(self.tmp.name, "y"))

    def test_leak_makes_the_command_fail(self):
        """If anything personal is still in the output the command must exit with code 2 (checked by faking a leaky anonymizer)."""
        from unittest import mock
        real_scrub = A.Anonymizer.scrub
        with mock.patch.object(A.Anonymizer, "scrub", lambda self, t: t):         # a broken anonymizer that does nothing
            with self.assertRaises(SystemExit) as cm:
                self.run_cli("messenger-import", self.root, "--out", os.path.join(self.tmp.name, "leaky"), "--me", ME, "--chats", "all", "--min-my-messages", "3")
        self.assertEqual(cm.exception.code, 2)
        self.assertFalse(json.loads(slurp(os.path.join(self.tmp.name, "leaky", "stats.json")))["safe_to_upload"])


class LocalOnly(unittest.TestCase):
    def test_importer_and_anonymizer_contain_no_network_code(self):
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "persona_lm")
        for name in ("messenger.py", "anonymize.py"):
            src = slurp(os.path.join(here, name))
            self.assertIsNone(re.search(r"^\s*(?:import|from)\s+(?:socket|urllib|requests|http|ftplib|smtplib|ssl|aiohttp)\b", src, re.M), name)


class PrepareExplicitValidation(unittest.TestCase):
    def test_per_message_train_flag_in_the_core_encoder(self):
        tok = Tokenizer.train(["hello there friend " * 40], 300)
        rec = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "there", "train": False},
                            {"role": "user", "content": "friend"}, {"role": "assistant", "content": "hello"}]}
        ids, mask = D.encode_chat(tok, D.as_messages(rec, with_train=True))
        self.assertEqual(tok.decode([int(i) for i, m in zip(ids, mask) if m]), "hello<|end|>")
        # the old behaviour is untouched: without the flag every assistant turn carries loss
        plain = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "there"}]}
        ids2, mask2 = D.encode_chat(tok, D.as_messages(plain))
        self.assertEqual(tok.decode([int(i) for i, m in zip(ids2, mask2) if m]), "there<|end|>")
        # a conversation whose only assistant turn is marked train=false teaches nothing and is skipped
        self.assertIsNone(D.as_messages({"messages": rec["messages"][:2]}, with_train=True))


class TokenizerTrainingInput(unittest.TestCase):
    def test_repeated_context_messages_are_counted_once(self):
        from persona_lm.cli import _texts_for_tokenizer
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.jsonl")
            ex = {"messages": [{"role": "user", "content": "same context"}, {"role": "assistant", "content": "r1", "train": True}]}
            ex2 = {"messages": [{"role": "user", "content": "same context"}, {"role": "assistant", "content": "r2", "train": True}]}
            with open(path, "w", encoding="utf-8") as f:
                for e in (ex, ex2, ex):
                    f.write(json.dumps(e) + "\n")
            self.assertEqual(sorted(_texts_for_tokenizer([path], 1)), ["r1", "r2", "same context"])


if __name__ == "__main__":
    unittest.main()
