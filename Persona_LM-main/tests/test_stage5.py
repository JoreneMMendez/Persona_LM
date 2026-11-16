"""Stage 5 tests: the emotion tagger and its learner, the mood, the Person card, their chat commands and command lines.
Nothing here reads the real keyboard or uses the network (a module guard makes any accidental keyboard read fail instead of wait)."""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import chatrepl, cli
from persona_lm import emotion as E
from persona_lm import memory as M
from persona_lm import person as P
from persona_lm.bpe import END, MEMORY, MOOD, ROLE_ID, SPEAKER, Tokenizer

HOUR = 3600.0
T0 = 1_700_000_000.0


def setUpModule():
    global _guard
    def no_keyboard(prompt=""):
        raise AssertionError("a test tried to read from the keyboard: %r" % prompt)
    _guard = mock.patch("builtins.input", no_keyboard)
    _guard.start()


def tearDownModule():
    _guard.stop()


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, hours):
        self.t += hours * HOUR


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.clock = Clock()

    def p(self, *parts):
        return os.path.join(self.root, *parts)


# a few sentences per feeling, written by hand (the learner tests need data with a known answer)
SAMPLE_WORDS = {
    "joy": ["happy", "excited", "masaya", "yay", "awesome", "malipay"],
    "love": ["love", "miss", "mahal", "palangga", "sweet", "hug"],
    "amusement": ["lol", "charing", "hahaha", "funny", "lingaw", "joke"],
    "sadness": ["sad", "iyak", "lungkot", "lonely", "naguol", "cry"],
    "anger": ["galit", "inis", "angry", "bwisit", "mad", "nasuko"],
    "tired": ["pagod", "antok", "kapoy", "sleepy", "puyat", "tired"],
    "neutral": ["table", "meeting", "tomorrow", "bus", "lunch", "report"],
}
FILLERS = ["today", "ako", "na", "kayo", "the", "daw", "ngayon", "friend", "ok", "this", "that", "very", "ba"]


def synthetic_rows(per_label=24):
    rows, i = [], 0
    for label, words in SAMPLE_WORDS.items():
        for k in range(per_label):
            w1, w2 = words[k % len(words)], words[(k * 2 + 1) % len(words)]
            f = FILLERS[(k + i) % len(FILLERS)], FILLERS[(k * 3 + i) % len(FILLERS)]
            text = "%s %s %s %s" % (f[0], w1, f[1], w2)
            i += 1
            rows.append({"id": i, "text": text, "label": label, "confidence": 0.5, "fixed": "", "evidence": label != "neutral"})
    return rows


def importer_style_jsonl(path, mine, others=("hello there", "how are you")):
    """Records shaped like messenger-import's window examples: context turns with train false, the owner's reply with train true."""
    with open(path, "w", encoding="utf-8") as f:
        for n, text in enumerate(mine):
            msgs = [{"role": "system", "content": "You are Aria.", "train": False},
                    {"role": "user", "content": others[n % len(others)], "train": False},
                    {"role": "assistant", "content": "earlier reply %d" % (n % 3), "train": False},
                    {"role": "user", "content": others[(n + 1) % len(others)], "train": False},
                    {"role": "assistant", "content": text, "train": True}]
            f.write(json.dumps({"chat": "chat_001", "session": 1, "messages": msgs}, ensure_ascii=False) + "\n")


class FakeGen:
    def __init__(self, tok, text="sure thing"):
        self.tok, self.text, self.calls, self.kwargs = tok, text, [], []

    def __call__(self, ids, **kw):
        self.calls.append(list(ids))
        self.kwargs.append(kw)
        return self.tok.encode(self.text, allow_special=False)


# ------------------------------------------------------------------------------------------------------------------ the tagger
class TaggerTests(unittest.TestCase):
    def setUp(self):
        self.t = E.Tagger()

    def label(self, text):
        return self.t.tag(text).label

    def test_english_tagalog_bisaya_and_emoji(self):
        for text, want in (("I am so happy today!", "joy"), ("pagod na ako, antok na", "tired"), ("kapoy kaayo ugh", "tired"),
                           ("nakakainis, galit ako", "anger"), ("huhu so sad 😭", "sadness"), ("Love you ❤️", "love"),
                           ("😂😂", "amusement"), ("malipay kaayo ko", "joy"), ("nagulat ako, wow", "surprise"), ("hala!", "surprise")):
            self.assertEqual(self.label(text), want, text)

    def test_laughter_of_any_length_and_stretched_words(self):
        for text in ("haha", "hahahahaha", "hehe ok", "HAHAHA"):
            self.assertEqual(self.label(text), "amusement", text)
        self.assertEqual(self.label("soooo tired"), "tired")
        self.assertEqual(self.label("sooooo saaaad"), "sadness")
        self.assertEqual(self.label("he said ha"), "neutral")                 # one "he" or "ha" is just a word

    def test_a_negated_feeling_word_is_ignored_not_flipped(self):
        for text in ("I'm not happy", "hindi ako masaya", "dili ko malipay", "never sad"):
            tag = self.t.tag(text)
            self.assertEqual((tag.label, tag.confidence), ("neutral", 0.0), text)
            self.assertEqual(tag.negated, 1)

    def test_no_evidence_is_neutral_with_zero_confidence_and_one_weak_word_does_not_win(self):
        tag = self.t.tag("see you at the bus stop")
        self.assertEqual((tag.label, tag.confidence, tag.evidence), ("neutral", 0.0, []))
        self.assertEqual(self.label("grabe"), "neutral")                       # a single weak word is not enough
        self.assertEqual(self.label("okay"), "neutral")
        self.assertEqual(self.label(""), "neutral")

    def test_confidence_grows_with_evidence_and_stays_between_zero_and_one(self):
        one, two, three = (self.t.tag(x).confidence for x in ("happy", "happy yay", "happy yay excited"))
        self.assertTrue(0 < one < two < three < 1)
        for text in ("x" * 500, "😂" * 200, "happy sad angry tired scared " * 20, "\u0000\n\t"):
            c = self.t.tag(text).confidence
            self.assertTrue(0 <= c <= 1, text[:20])

    def test_mixed_feelings_give_a_lower_confidence_than_one_clear_feeling(self):
        self.assertLess(self.t.tag("happy sad").confidence, self.t.tag("happy happy").confidence)

    def test_case_apostrophes_and_the_same_answer_every_time(self):
        self.assertEqual(self.label("I AM ANGRY"), "anger")
        self.assertEqual(self.label("DON'T be ANGRY"), "neutral")                # negated: dont + angry (apostrophes dropped)
        self.assertEqual(self.t.tag("Pagod AKO").describe(), self.t.tag("pagod ako").describe())
        self.assertEqual(self.t.tag("happy").scores, self.t.tag("happy").scores)

    def test_words_that_mean_something_else_in_the_other_language_are_not_in_the_list(self):
        lookup = E.build_lookup()
        for w in ("suko", "sayang", "lagot", "gago"):
            self.assertNotIn(w, lookup, w)

    def test_every_label_has_coordinates_in_range_and_is_its_own_nearest_label(self):
        self.assertEqual(set(E.LABELS), set(E.LEXICON) | {"neutral"})
        for label, (v, a) in E.LABELS.items():
            self.assertTrue(-1 <= v <= 1 and -1 <= a <= 1)
            self.assertEqual(E.nearest_label(v, a), label)

    def test_lexicon_entries_are_lower_case_and_not_listed_twice_for_one_feeling(self):
        for label, (strong, weak) in E.LEXICON.items():
            words = strong + weak
            self.assertEqual(len(words), len(set(words)), label)
            for w in words:
                self.assertEqual(w, w.casefold(), (label, w))

    def test_extra_word_list_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = os.path.join(d, "lex.json")
            write(path, json.dumps({"joy": ["chix"], "anger": ["ngitngit"]}))
            t = E.Tagger(None, E.load_lexicon_file(path))
            self.assertEqual(t.tag("chix na chix").label, "joy")
            self.assertEqual(t.tag("ngitngit ako").label, "anger")
            self.assertEqual(E.Tagger().tag("chix").label, "neutral")
            for bad in ('{"joyy": ["x"]}', '{"joy": "x"}', "[1]", "{broken"):
                write(path, bad)
                with self.assertRaises(E.EmotionProblem):
                    E.load_lexicon_file(path)
            with self.assertRaises(E.EmotionProblem):
                E.load_lexicon_file(os.path.join(d, "missing.json"))

    def test_the_two_new_modules_need_no_network_and_no_heavy_libraries(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "persona_lm")
        for name in ("emotion.py", "person.py"):
            src = read(os.path.join(root, name))
            for bad in ("import torch", "numpy", "socket", "urllib", "requests", "http.client", "subprocess"):
                self.assertNotIn(bad, src, name + " " + bad)


# ------------------------------------------------------------------------------------------------------------------ the learner
class LearnerTests(Base):
    def test_learns_the_words_of_each_feeling_from_labelled_messages(self):
        rows = [(r["text"], r["label"], 1.0) for r in synthetic_rows()]
        m = E.NaiveBayes.train(rows, min_count=1)
        hits = sum(1 for r in synthetic_rows() if max(m.probabilities(r["text"]).items(), key=lambda kv: kv[1])[0] == r["label"])
        self.assertGreater(hits / len(rows), 0.9)

    def test_probabilities_add_up_to_one_and_unknown_words_change_nothing(self):
        m = E.NaiveBayes.train([(r["text"], r["label"], 1.0) for r in synthetic_rows()], min_count=1)
        p1 = m.probabilities("pagod today")
        self.assertAlmostEqual(sum(p1.values()), 1.0, places=9)
        p2 = m.probabilities("pagod today zzzunknownzzz qqqqqq")
        for l in E.LABEL_ORDER:
            self.assertAlmostEqual(p1[l], p2[l], places=9)

    def test_words_seen_only_once_are_dropped(self):
        rows = [("happy happy zorblax", "joy", 1.0), ("happy day", "joy", 1.0), ("sad day", "sadness", 1.0), ("sad night", "sadness", 1.0)]
        m = E.NaiveBayes.train(rows, min_count=2)
        known = set().union(*[set(fc) for fc in m.data["feature_counts"].values()])
        self.assertIn("w:happy", known)
        self.assertNotIn("w:zorblax", known)

    def test_your_corrections_count_more(self):
        rows = [("ay nako lagi", "neutral", 1.0)] * 4 + [("ay nako lagi", "sadness", 5.0)]
        m = E.NaiveBayes.train(rows, min_count=1)
        p = m.probabilities("ay nako lagi")
        self.assertGreater(p["sadness"], p["neutral"])

    def test_unknown_feeling_in_the_training_data_is_refused(self):
        with self.assertRaises(E.EmotionProblem):
            E.NaiveBayes.train([("x", "hungry", 1.0)])

    def test_save_load_round_trip_with_one_backup_and_an_atomic_write(self):
        m = E.NaiveBayes.train([(r["text"], r["label"], 1.0) for r in synthetic_rows()], min_count=1)
        path = self.p("sub", "model.json")
        m.save(path)
        self.assertFalse(os.path.exists(path + ".tmp"))
        m.save(path)
        self.assertTrue(os.path.exists(self.p("sub", "model.prev.json")))
        m2 = E.NaiveBayes.load(path)
        for text in ("pagod today", "galit na ako", "xyz"):
            self.assertEqual(m.probabilities(text), m2.probabilities(text))

    def test_a_broken_or_foreign_model_file_is_refused_with_a_clear_message(self):
        for body in ("{broken", "[1, 2]", '{"kind": "something-else"}', '{"kind": "naive-bayes", "format_version": 99}'):
            write(self.p("m.json"), body)
            with self.assertRaises(E.EmotionProblem):
                E.NaiveBayes.load(self.p("m.json"))
        with self.assertRaises(E.EmotionProblem):
            E.NaiveBayes.load(self.p("nope.json"))

    def test_the_learner_decides_when_there_is_no_known_feeling_word(self):
        rows = [("charing lang ako ngayon", "amusement", 1.0)] * 6 + [("charing ra man", "amusement", 1.0)] * 6 + \
               [("meeting tomorrow at the bus", "neutral", 1.0)] * 12
        model = E.NaiveBayes.train(rows, min_count=2)
        self.assertEqual(E.Tagger().tag("charing lang").label, "neutral")                       # the word list does not know "charing" ...
        self.assertEqual(E.Tagger(model).tag("charing lang").label, "amusement")                # ... but the learner learned it from your messages

    def test_word_list_and_learner_agreeing_make_the_tagger_more_sure(self):
        rows = [(r["text"], r["label"], 1.0) for r in synthetic_rows()]
        model = E.NaiveBayes.train(rows, min_count=1)
        text = "ako pagod na pagod"
        self.assertGreater(E.Tagger(model).tag(text).confidence, E.Tagger().tag(text).confidence)

    def test_training_report_is_honest_about_what_it_measures(self):
        rows = synthetic_rows()
        for r in rows[:20]:
            r["fixed"] = r["label"]
        model, rep = E.train_from_samples(rows, min_count=1)
        self.assertEqual((rep["messages"], rep["fixed"]), (len(rows), 20))
        self.assertGreater(rep["held_out"], 0)
        self.assertIn("agrees_with_word_list", rep)
        self.assertEqual(model.data["trained_on"], {"messages": len(rows), "owner_corrected": 20})
        rows2 = synthetic_rows()
        _, rep2 = E.train_from_samples(rows2, min_count=1)
        self.assertNotIn("fixed_right_with_model", rep2)                # no corrections -> no claim about real accuracy


# ------------------------------------------------------------------------------------------------------------------ samples and review
class SampleTests(Base):
    def test_only_your_own_messages_with_loss_on_are_taken_once_each(self):
        path = self.p("train.jsonl")
        importer_style_jsonl(path, ["hello po", "pagod na ako", "hello po"])
        got = E.owner_messages([path])
        self.assertEqual(got, ["hello po", "pagod na ako"])
        self.assertNotIn("earlier reply 0", got)
        self.assertNotIn("hello there", got)

    def test_neutral_messages_are_kept_only_in_proportion(self):
        t = E.Tagger()
        texts = ["happy %d" % i for i in range(10)] + ["see you at table %d" % i for i in range(200)]
        rows = E.make_samples(texts, t, neutral_ratio=3, seed=5)
        self.assertEqual(sum(1 for r in rows if r["label"] != "neutral"), 10)
        self.assertEqual(sum(1 for r in rows if r["label"] == "neutral"), 30)
        self.assertEqual(sorted(r["id"] for r in rows), list(range(1, 41)))
        self.assertEqual(rows, E.make_samples(texts, t, neutral_ratio=3, seed=5))     # same seed -> same file
        self.assertTrue(all(r["fixed"] == "" for r in rows))

    def test_very_long_messages_are_skipped(self):
        rows = E.make_samples(["happy " + "x" * 2000, "happy"], E.Tagger(), neutral_ratio=None)
        self.assertEqual([r["text"] for r in rows], ["happy"])

    def test_the_samples_file_is_never_overwritten_by_mistake(self):
        rows = E.make_samples(["happy"], E.Tagger())
        E.write_samples(self.p("s.jsonl"), rows)
        with self.assertRaises(E.EmotionProblem) as cm:
            E.write_samples(self.p("s.jsonl"), rows)
        self.assertIn("already exists", str(cm.exception))
        self.assertFalse(os.path.exists(self.p("s.jsonl.tmp")))

    def test_a_bad_samples_file_is_refused(self):
        for body in ('{"text": "x", "label": "hungry"}\n', '{"label": "joy"}\n', '{"text": "x", "label": "joy", "fixed": "zzz"}\n', "not json\n"):
            write(self.p("s.jsonl"), body)
            with self.assertRaises(E.EmotionProblem):
                E.read_samples(self.p("s.jsonl"))

    def make_file(self):
        rows = [{"id": i, "text": t, "label": l, "confidence": 0.5, "fixed": "", "evidence": e} for i, (t, l, e) in enumerate(
            [("happy day", "joy", True), ("so tired", "tired", True), ("grabe", "neutral", True), ("see you", "neutral", False), ("galit ako", "anger", True)], 1)]
        E.write_samples(self.p("s.jsonl"), rows)

    def typed(self, answers):
        it = iter(answers)
        seen = []
        def fn(prompt):
            seen.append(prompt)
            return next(it)
        return fn, seen

    def test_review_accepts_corrects_skips_and_saves_every_answer_at_once(self):
        self.make_file()
        fn, _ = self.typed(["", "sadness", "s", "joy"])
        shown = []
        n = E.review(self.p("s.jsonl"), fn, out=shown.append, count=10, seed=3)
        rows = {r["text"]: r for r in E.read_samples(self.p("s.jsonl"))}
        self.assertEqual(n, 3)                                           # 4 messages offered (the evidence-free neutral one is not), one skipped
        answered = {t: r["fixed"] for t, r in rows.items() if r["fixed"]}
        self.assertEqual(len(answered), 3)
        self.assertNotIn("see you", answered)
        self.assertIn("sadness", answered.values())
        for text, r in rows.items():                                     # the guess stays in "label"; the answer is in "fixed"
            self.assertIn(r["label"], E.LABELS)

    def test_review_asks_again_for_a_wrong_word_and_quit_keeps_what_was_answered(self):
        self.make_file()
        fn, seen = self.typed(["happyish", "joy", "q"])
        shown = []
        E.review(self.p("s.jsonl"), fn, out=shown.append, count=10, seed=3)
        self.assertEqual(len(seen), 3)
        self.assertTrue(any("not a feeling" in s for s in shown))
        self.assertEqual(sum(1 for r in E.read_samples(self.p("s.jsonl")) if r["fixed"]), 1)

    def test_review_stops_cleanly_when_the_input_ends_and_never_re_asks_answered_ones(self):
        self.make_file()
        def fn(prompt):
            raise EOFError
        self.assertEqual(E.review(self.p("s.jsonl"), fn, out=lambda *_: None), 0)
        fn2, _ = self.typed([""] * 10)
        E.review(self.p("s.jsonl"), fn2, out=lambda *_: None, count=10)
        shown = []
        self.assertEqual(E.review(self.p("s.jsonl"), fn2, out=shown.append, count=10), 0)
        self.assertIn("Nothing left", shown[0])


# ------------------------------------------------------------------------------------------------------------------ the mood
class MoodTests(Base):
    def mood(self, path=None):
        return E.Mood(path, clock=self.clock)

    def test_messages_move_the_mood_toward_their_feeling(self):
        m, t = self.mood(), E.Tagger()
        v0 = m.valence
        m.update(t.tag("so happy yay"), self.clock())
        self.assertGreater(m.valence, v0)
        m2 = self.mood()
        for _ in range(4):
            m2.update(t.tag("galit na galit ako, bwisit"), self.clock())
        self.assertLess(m2.valence, 0)
        self.assertGreater(m2.arousal, 0.3)
        self.assertEqual(m2.label, "anger")

    def test_tired_messages_lower_arousal_and_sad_ones_lower_valence(self):
        m = self.mood()
        for _ in range(4):
            m.update(E.Tagger().tag("pagod na pagod, antok"), self.clock())
        self.assertLess(m.arousal, -0.3)
        self.assertEqual(m.label, "tired")

    def test_neutral_or_unsure_messages_do_not_move_it_and_unsure_moves_it_less(self):
        m = self.mood()
        before = (m.valence, m.arousal)
        m.update(E.Tagger().tag("see you at the bus"), self.clock())
        self.assertEqual((m.valence, m.arousal), before)
        a, b = self.mood(), self.mood()
        a.update(E.Tagger().tag("happy"), self.clock())
        b.update(E.Tagger().tag("happy yay excited"), self.clock())
        self.assertLess(a.valence, b.valence)

    def test_values_always_stay_between_minus_one_and_one(self):
        m, t = self.mood(), E.Tagger()
        for text in ("galit " * 50, "love ❤️ " * 50, "pagod " * 50) * 20:
            m.update(t.tag(text), self.clock())
            self.assertTrue(-1 <= m.valence <= 1 and -1 <= m.arousal <= 1)

    def test_it_drifts_back_to_the_baseline_with_a_six_hour_half_life(self):
        m = self.mood()
        for _ in range(5):
            m.update(E.Tagger().tag("galit bwisit inis"), self.clock())
        d0 = ((m.valence - m.baseline[0]) ** 2 + (m.arousal - m.baseline[1]) ** 2) ** 0.5
        self.clock.advance(6)
        m.update(E.Tag("neutral", 0.0, {}, []), self.clock())
        d1 = ((m.valence - m.baseline[0]) ** 2 + (m.arousal - m.baseline[1]) ** 2) ** 0.5
        self.assertAlmostEqual(d1 / d0, 0.5, places=6)
        self.clock.advance(24 * 3)
        m.update(E.Tag("neutral", 0.0, {}, []), self.clock())
        self.assertAlmostEqual(m.valence, m.baseline[0], places=3)
        self.assertEqual(m.label, E.nearest_label(*m.baseline))

    def test_a_clock_that_goes_backwards_does_not_blow_up(self):
        m = self.mood()
        m.update(E.Tagger().tag("happy"), self.clock())
        v = m.valence
        m.update(E.Tag("neutral", 0.0, {}, []), self.clock() - 10 * HOUR)
        self.assertAlmostEqual(m.valence, v, places=9)

    def test_prompt_text_has_a_fixed_shape_and_temperature_factor_is_small(self):
        m = self.mood()
        self.assertRegex(m.prompt_text(), r"^[a-z]+ [+-]\d\.\d [+-]\d\.\d$")
        for v, a in ((-1, -1), (1, 1), (0, 0), (0.3, -0.7)):
            m.valence, m.arousal = v, a
            self.assertRegex(m.prompt_text(), r"^[a-z]+ [+-]\d\.\d [+-]\d\.\d$")
            self.assertTrue(0.85 <= m.temperature_factor() <= 1.15)

    def test_save_load_round_trip_and_a_broken_file_changes_nothing(self):
        path = self.p("sub", "mood.json")
        m = self.mood(path)
        m.update(E.Tagger().tag("so tired, pagod"), self.clock())
        m.save()
        self.assertFalse(os.path.exists(path + ".tmp"))
        again = self.mood(path)
        self.assertEqual((again.valence, again.arousal, again.turns, again.history), (round(m.valence, 4), round(m.arousal, 4), m.turns, m.history))
        for body in ("{broken", "[]", '{"valence": "x"}', '{"valence": 0, "arousal": 0}'):
            write(path, body)
            with self.assertRaises(E.EmotionProblem):
                self.mood(path)
            self.assertEqual(read(path), body)

    def test_snapshot_restore_reset_and_history_cap(self):
        m, t = self.mood(), E.Tagger()
        snap = m.snapshot()
        m.update(t.tag("galit"), self.clock())
        m.restore(snap)
        self.assertEqual(m.snapshot(), snap)
        for _ in range(80):
            m.update(t.tag("happy yay"), self.clock())
        self.assertEqual(len(m.history), E.Mood.MAX_HISTORY)
        m.reset()
        self.assertEqual((m.valence, m.arousal), tuple(m.baseline))


# ------------------------------------------------------------------------------------------------------------------ the chat screen
class ChatMoodTests(Base):
    def setUp(self):
        super().setUp()
        corpus = "the cat sat on the mat. maria loves mango and the dog. hello how are you today fine thanks name card speaking style " * 40
        self.tok = Tokenizer.train([corpus], 300)

    def session(self, text="sure thing", mood=True, person=None, memory=None, ctx=256, **kw):
        self.gen = FakeGen(self.tok, text)
        m = E.Mood(self.p("mood.json"), clock=self.clock) if mood else None
        return chatrepl.ChatSession(self.tok, ctx, self.gen, save_dir=self.root, clock=self.clock, mood=m, person=person, memory=memory, **kw)

    def blocks(self, ids):
        """{special id: decoded text} for the blocks in a prompt"""
        out, i = {}, 0
        while i < len(ids):
            if ids[i] in (MOOD, SPEAKER, MEMORY):
                j = ids.index(END, i)
                out[ids[i]] = self.tok.decode(ids[i + 1:j])
                i = j
            i += 1
        return out

    def test_commands_explain_themselves_when_mood_is_off(self):
        s = self.session(mood=False)
        text, _ = s.handle("/mood")
        self.assertIn("--mood", text)
        s.handle("hello")
        self.assertNotIn(MOOD, self.gen.calls[0])
        self.assertEqual(os.listdir(self.root), [])

    def test_the_prompt_carries_one_mood_block_after_the_system_line(self):
        s = self.session(system="You are Aria.")
        s.handle("hello")
        ids = self.gen.calls[0]
        self.assertEqual(ids.count(MOOD), 1)
        self.assertLess(ids.index(ROLE_ID["system"]), ids.index(MOOD))
        self.assertLess(ids.index(MOOD), ids.index(ROLE_ID["user"]))
        self.assertRegex(self.blocks(ids)[MOOD], r"^[a-z]+ [+-]\d\.\d [+-]\d\.\d$")

    def test_what_you_type_moves_the_mood_and_the_ai_sees_it_in_the_same_turn(self):
        s = self.session()
        for _ in range(3):
            s.handle("pagod na pagod ako, antok")
        self.assertEqual(s.mood.label, "tired")
        self.assertTrue(self.blocks(self.gen.calls[-1])[MOOD].startswith("tired "))
        self.assertIn("feeling now: tired", s.handle("pagod")[0])

    def test_the_models_own_words_never_change_the_mood_or_run_commands(self):
        s = self.session(text="galit na galit ako!!! bwisit /mood reset /quit")
        for _ in range(3):
            text, quit_ = s.handle("hello")
            self.assertFalse(quit_)
        base = E.Mood(None, clock=self.clock)
        self.assertEqual((s.mood.valence, s.mood.arousal), (base.valence, base.arousal))

    def test_a_refused_message_or_an_empty_reply_leaves_the_mood_alone(self):
        s = self.session(text="  ")
        before = s.mood.snapshot()
        text, _ = s.handle("galit na galit bwisit " + "kayo " * 400)
        self.assertIn("too long", text)
        self.assertEqual(s.mood.snapshot(), before)
        self.assertFalse(os.path.exists(self.p("mood.json")))
        text, _ = s.handle("galit na galit bwisit")
        self.assertIn("empty reply", text)
        self.assertEqual(s.mood.snapshot(), before)
        self.assertEqual(s.history, [])

    def test_mood_is_saved_after_a_reply_and_the_next_chat_picks_it_up(self):
        s = self.session()
        s.handle("sad na sad ako, iyak")
        self.assertTrue(os.path.exists(self.p("mood.json")))
        s2 = self.session()
        self.assertEqual(s2.mood.label, s.mood.label)
        self.assertEqual(s2.mood.turns, 1)

    def test_mood_command_shows_tags_without_changing_and_resets(self):
        s = self.session()
        s.handle("pagod na pagod")
        before = s.mood.snapshot()
        shown = s.handle("/mood")[0]
        self.assertIn("Feeling: ", shown)
        self.assertIn("last message was read as: tired", shown)
        self.assertIn("<|mood|>", shown)
        self.assertIn("anger", s.handle("/mood tag galit na galit ako")[0])
        self.assertEqual(s.mood.snapshot(), before)
        self.assertIn("Usage", s.handle("/mood tag")[0])
        self.assertIn("Usage", s.handle("/mood dance")[0])
        self.assertIn("reset", s.handle("/mood reset")[0])
        self.assertEqual((s.mood.valence, s.mood.arousal), tuple(s.mood.baseline))
        self.assertEqual(len(self.gen.calls), 1)                     # none of this reached the model

    def test_temperature_follows_arousal_but_the_users_setting_is_untouched(self):
        s = self.session()
        s.mood.valence, s.mood.arousal = 0.0, 0.8
        s.mood.updated = self.clock()
        s.handle("see you at the bus")
        self.assertAlmostEqual(self.gen.kwargs[-1]["temperature"], 0.8 * 1.12, places=3)
        self.assertEqual(s.settings["temperature"], 0.8)
        s.handle("/temp 0")
        s.handle("see you again")
        self.assertEqual(self.gen.kwargs[-1]["temperature"], 0)
        s2 = self.session(mood=False)
        s2.handle("hi")
        self.assertEqual(self.gen.kwargs[-1]["temperature"], 0.8)

    def card(self, extra=()):
        c = P.new_card("Jaymar")
        c.add_owner("Values", "I care about my family first")
        c.add_owner("Habits", "I stay up late")
        for t in extra:
            c.add_owner("Habits", t)
        s = c.section("Relationships")
        s[1].insert(0, {"kind": "fact", "tag": "auto", "text": "Maria: Maria is my sister (2 memories)"})
        s[1].insert(0, {"kind": "fact", "tag": "no", "text": "Jun: never put this in a prompt"})
        return c

    def test_the_person_card_goes_in_a_speaker_block_before_the_mood_block(self):
        s = self.session(system="You are Aria.", person=self.card(), ctx=1024, person_tokens=300)
        s.handle("hello")
        ids = self.gen.calls[0]
        self.assertEqual(ids.count(SPEAKER), 1)
        self.assertLess(ids.index(ROLE_ID["system"]), ids.index(SPEAKER))
        self.assertLess(ids.index(SPEAKER), ids.index(MOOD))
        text = self.blocks(ids)[SPEAKER]
        self.assertTrue(text.startswith("Name: Jaymar"))
        self.assertIn("Maria", text)
        self.assertIn("family", text)
        self.assertNotIn("never put this", text)

    def test_the_card_never_gets_cut_in_the_middle_of_a_line_and_stays_inside_its_budget(self):
        card = self.card(["I wake up at %d and drink coffee with milk" % n for n in range(30)])
        lines_all = {l for l in P.prompt_text(card, lambda s: 0, 10**9)[0].split("\n")}
        s = self.session(person=card, person_tokens=40, ctx=512)
        shown = s.handle("hello")[0]
        block = self.blocks(self.gen.calls[0])[SPEAKER]
        self.assertLessEqual(len(self.tok.encode(block, allow_special=False)) + 2, 40)
        for line in block.split("\n"):
            self.assertIn(line, lines_all)
        self.assertGreater(len(lines_all), len(block.split("\n")))
        self.assertIn("did not fit the card budget", shown)
        self.assertNotIn("did not fit the card budget", s.handle("again")[0])       # said once

    def test_everything_on_with_a_small_window_still_fits_and_keeps_the_newest_message(self):
        store = M.MemoryStore(None, clock=self.clock)
        store.add("[[person:Maria]] loves mango", importance=0.9)
        s = self.session(person=self.card(["I wake up at %d" % n for n in range(30)]), memory=store, system="You are Aria.", ctx=128, settings={"max_new": 16})
        for text in ("hello Maria", "tell me about mango", "and the dog?"):
            reply = s.handle(text)[0]
            self.assertTrue(reply.startswith("ai> "), reply)
            ids = self.gen.calls[-1]
            self.assertLessEqual(len(ids), s.prompt_budget())
            self.assertIn(self.tok.encode(text, allow_special=False)[0], ids)
            self.assertEqual(ids.count(MOOD), 1)

    def test_order_is_system_then_speaker_then_mood_then_memory(self):
        store = M.MemoryStore(None, clock=self.clock)
        store.add("Maria loves mango", importance=0.9)
        s = self.session(person=self.card(), memory=store, system="You are Aria.")
        s.handle("what does Maria like")
        ids = self.gen.calls[0]
        order = [ids.index(x) for x in (ROLE_ID["system"], SPEAKER, MOOD, MEMORY, ROLE_ID["user"])]
        self.assertEqual(order, sorted(order))

    def test_no_card_means_no_speaker_block(self):
        s = self.session()
        s.handle("hello")
        self.assertNotIn(SPEAKER, self.gen.calls[0])


# ------------------------------------------------------------------------------------------------------------------ the Person card
class CardTests(Base):
    def test_new_card_renders_and_parses_back_to_the_same_text(self):
        c = P.new_card("Jaymar")
        text = c.render()
        self.assertEqual(P.Card.parse(text).render(), text)
        self.assertEqual([t for t, _ in P.Card.parse(text).sections], P.SECTIONS)
        self.assertEqual([f["text"] for f in P.Card.parse(text).facts("Name")], ["Jaymar"])

    def test_a_hand_edited_file_survives_a_round_trip_with_comments_unknown_sections_and_emoji(self):
        text = ("# Person card\nmy own note\n\n## Name\n- Jaymar 😂\n\n## Hobbies\n- basketball\nfree text I typed\n- [auto] x: y\n- [no] Emoji: stop\n"
                "\n## Values\n<!-- thinking about this -->\n- family first\n")
        c = P.Card.parse(text)
        self.assertEqual(c.render(), text)
        self.assertEqual([(f["tag"], f["text"]) for f in c.facts("Hobbies")], [("owner", "basketball"), ("auto", "x: y"), ("no", "Emoji: stop")])
        self.assertEqual(c.facts("Values")[0]["tag"], "owner")                   # no tag means yours

    def test_add_goes_after_the_last_fact_checks_the_section_and_the_length(self):
        c = P.new_card("Jaymar")
        c.add_owner("values", "family first")
        c.add_owner("Values", "honesty")
        self.assertEqual([f["text"] for f in c.facts("Values")], ["family first", "honesty"])
        self.assertIn("## Values\n- family first\n- honesty\n", c.render())
        for bad, msg in (("", "empty"), ("x" * 401, "short")):
            with self.assertRaises(P.PersonProblem) as cm:
                c.add_owner("Values", bad)
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(P.PersonProblem) as cm:
            c.add_owner("Hobbies", "x")
        self.assertIn("Sections:", str(cm.exception))

    def test_a_missing_card_gives_a_message_that_says_how_to_make_one(self):
        with self.assertRaises(P.PersonProblem) as cm:
            P.load(self.p("nope.md"))
        self.assertIn("person init", str(cm.exception))

    def messages(self):
        return ["Hello there friend."] * 10 + ["haha ok lang"] * 10 + ["sooo tired 😂"] * 10

    def test_style_numbers_are_plain_counts_of_the_messages(self):
        facts = dict(P.style_facts(self.messages()))
        self.assertIn("30 messages analysed", facts["Length"])
        self.assertIn("67%", facts["Capital letters"])                    # 20 of 30 start with a lower-case letter... "sooo" and "haha"
        self.assertIn("33%", facts["Punctuation"])                        # 10 of 30 end with a full stop
        self.assertIn("33%", facts["Laughter"])
        self.assertIn("😂 (10)", facts["Emoji"])
        self.assertIn("33%", facts["Stretched words"])
        self.assertEqual(P.style_facts(["hi"] * 19), [])                  # too few messages to describe a style

    def test_placeholders_laughter_and_stretched_words_are_not_topics(self):
        texts = ["[ME] Person_01 project haha sooo project"] * 6
        words = dict(P.frequent_words(texts))["Top words"]
        self.assertIn("project (6)", words)
        for bad in ("haha", "sooo", "person", "[me]"):
            self.assertNotIn(bad, words.lower())
        self.assertEqual(P.frequent_words(["project"] * 3), [])           # under five messages is not a pattern

    def test_relationships_come_from_active_person_nodes_only(self):
        st = M.MemoryStore(None, clock=self.clock)
        st.add("[[person:Maria]] is my sister. She likes mango.", emotion="love", importance=0.9)
        st.add("[[person:Maria]] called today", emotion="joy")
        st.add("[[topic:Cebu]] trip is next month")
        gone = st.add("[[person:Jun]] used to be my classmate")
        gone["status"] = "archived"
        facts = P.relationship_facts(st)
        self.assertEqual([k for k, _ in facts], ["Maria"])
        self.assertIn("Maria is my sister.", facts[0][1])
        self.assertIn("2 memories, mostly tagged", facts[0][1])

    def build_card(self):
        c = P.new_card("Jaymar")
        st = M.MemoryStore(None, clock=self.clock)
        st.add("[[person:Maria]] is my sister", emotion="love")
        return c, st

    def test_a_plan_changes_nothing_and_apply_adds_only_auto_lines(self):
        c, st = self.build_card()
        before = c.render()
        plan = P.plan_build(c, self.messages(), st)
        self.assertEqual(c.render(), before)
        self.assertTrue(plan["add"] and not plan["remove"])
        P.apply_plan(c, plan)
        after = c.render()
        self.assertIn("- [auto] Maria: Maria is my sister", after)
        self.assertIn("- [auto] Emoji:", after)
        self.assertIn("- Jaymar\n", after)

    def test_building_twice_with_the_same_data_changes_nothing_the_second_time(self):
        c, st = self.build_card()
        P.apply_plan(c, P.plan_build(c, self.messages(), st))
        again = P.plan_build(c, self.messages(), st)
        self.assertEqual(again, {"add": [], "remove": []})

    def test_a_rebuild_replaces_old_auto_lines_with_fresh_numbers(self):
        c, st = self.build_card()
        P.apply_plan(c, P.plan_build(c, self.messages(), st))
        more = self.messages() + ["plain words here"] * 30
        plan = P.plan_build(c, more, st)
        self.assertTrue(any(t.startswith("Length:") for _, t in plan["remove"]))
        self.assertTrue(any(t.startswith("Length:") for _, t in plan["add"]))
        P.apply_plan(c, plan)
        self.assertEqual(sum(1 for f in c.facts("Speaking style", ("auto",)) if f["text"].startswith("Length:")), 1)
        self.assertIn("60 messages analysed", c.render())

    def test_lines_you_wrote_are_never_touched_and_a_line_you_wrote_about_a_key_blocks_the_auto_one(self):
        c, st = self.build_card()
        c.add_owner("Speaking style", "Length: I write long messages when I am happy")
        c.add_owner("Relationships", "Maria: my little sister, we talk every day")
        mine = [f["text"] for s in ("Speaking style", "Relationships") for f in c.facts(s, ("owner",))]
        P.apply_plan(c, P.plan_build(c, self.messages(), st))
        self.assertEqual([f["text"] for s in ("Speaking style", "Relationships") for f in c.facts(s, ("owner",))], mine)
        self.assertFalse(any(f["text"].startswith(("Length:", "Maria:")) for s in ("Speaking style", "Relationships") for f in c.facts(s, ("auto",))))

    def test_a_no_line_stops_that_suggestion_for_good_and_stays_in_the_file(self):
        c, st = self.build_card()
        c.section("Speaking style")[1].insert(0, {"kind": "fact", "tag": "no", "text": "Emoji: do not describe my emoji"})
        P.apply_plan(c, P.plan_build(c, self.messages(), st))
        P.apply_plan(c, P.plan_build(c, self.messages() * 2, st))
        self.assertFalse(any(f["text"].startswith("Emoji:") for f in c.facts("Speaking style", ("auto",))))
        self.assertEqual(len(c.facts("Speaking style", ("no",))), 1)

    def test_an_auto_line_you_keep_by_deleting_the_tag_becomes_yours(self):
        c, st = self.build_card()
        P.apply_plan(c, P.plan_build(c, self.messages(), st))
        text = c.render().replace("- [auto] Maria:", "- Maria:")
        c2 = P.Card.parse(text)
        P.apply_plan(c2, P.plan_build(c2, self.messages(), st))
        self.assertIn("- Maria: Maria is my sister", c2.render())
        self.assertEqual(c2.render().count("Maria:"), 1)

    def test_a_section_without_data_is_left_alone(self):
        c, st = self.build_card()
        P.apply_plan(c, P.plan_build(c, self.messages(), st))
        before = c.render()
        plan = P.plan_build(c, None, st)                                  # no messages given: Speaking style / Frequent words stay as they are
        self.assertEqual(plan["remove"], [])
        plan = P.plan_build(c, self.messages(), None)                     # no memory file given: Relationships stay
        self.assertFalse(any(s == "Relationships" for s, _ in plan["add"] + plan["remove"]))
        self.assertEqual(c.render(), before)

    def test_habits_and_values_are_never_built(self):
        c, st = self.build_card()
        plan = P.plan_build(c, self.messages(), st)
        self.assertFalse(any(s in ("Habits", "Values", "Name") for s, _ in plan["add"] + plan["remove"]))

    def test_prompt_text_order_budget_and_whole_lines(self):
        c = P.new_card("Jaymar")
        c.add_owner("Habits", "I stay up late")
        c.add_owner("Values", "family first")
        c.section("Relationships")[1].insert(0, {"kind": "fact", "tag": "auto", "text": "Maria: my sister"})
        c.section("Relationships")[1].insert(0, {"kind": "fact", "tag": "no", "text": "Jun: banned"})
        text, used, left = P.prompt_text(c, lambda s: len(s), 10**6)
        self.assertEqual(text.split("\n"), ["Name: Jaymar", "Relationships: Maria: my sister", "Values: family first", "Habits: I stay up late"])
        self.assertEqual((used, left), (4, 0))
        text, used, left = P.prompt_text(c, lambda s: len(s), 40)
        self.assertEqual(text.split("\n")[0], "Name: Jaymar")
        self.assertLessEqual(sum(len(l) + 1 for l in text.split("\n")), 40)
        self.assertEqual(used + left, 4)
        self.assertGreater(left, 0)
        self.assertEqual(P.prompt_text(c, lambda s: len(s), 0)[1:], (0, 4))


# ------------------------------------------------------------------------------------------------------------------ command lines
class CommandLineTests(Base):
    def run_cli(self, *argv, typed=None):
        buf = io.StringIO()
        ctx = mock.patch("builtins.input", lambda prompt="": next(typed)) if typed is not None else contextlib.nullcontext()
        with ctx, contextlib.redirect_stdout(buf):
            cli.main(list(argv))
        return buf.getvalue()

    def fails(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            cli.main(list(argv))
        return str(cm.exception)

    def chats(self, n=60):
        mine = []
        for i in range(n):
            mine.append(["happy na happy ako, yay %d", "pagod na pagod, antok %d", "galit ako, bwisit %d", "haha sooo funny 😂 %d", "see you at the bus %d", "okay po %d"][i % 6] % i)
        importer_style_jsonl(self.p("train.jsonl"), mine)
        return self.p("train.jsonl")

    def test_person_card_workflow(self):
        card = self.p("card.md")
        self.run_cli("person", "--file", card, "init", "--name", "Jaymar")
        self.assertIn("already exists", self.fails("person", "--file", card, "init"))
        self.assertIn("## Values", self.run_cli("person", "--file", card, "show"))
        self.run_cli("person", "--file", card, "add", "Values", "family", "first")
        train = self.chats()
        mem = self.p("mem.json")
        self.run_cli("memory", "--file", mem, "add", "[[person:Maria]]", "is", "my", "sister", "#love")
        before = read(card)
        out = self.run_cli("person", "--file", card, "build", "--data", train, "--memory", mem)
        self.assertIn("nothing changes unless you add --apply", out)
        self.assertIn("+ [Relationships] Maria:", out)
        self.assertEqual(read(card), before)
        self.run_cli("person", "--file", card, "build", "--data", train, "--memory", mem, "--apply")
        self.assertTrue(os.path.exists(self.p("card.prev.md")))
        self.assertEqual(read(self.p("card.prev.md")), before)
        text = read(card)
        self.assertIn("- family first", text)
        self.assertIn("- [auto] Maria:", text)
        self.assertIn("Emoji:", text)
        self.assertIn("nothing to change", self.run_cli("person", "--file", card, "build", "--data", train, "--memory", mem))
        self.assertIn("Give --data", self.fails("person", "--file", card, "build"))
        self.assertIn("Unknown section", self.fails("person", "--file", card, "add", "Hobbies", "x"))

    def test_emotion_workflow_label_review_train_tag_report(self):
        train = self.chats()
        out = self.run_cli("emotion", "label", "--data", train, "--out", self.p("s.jsonl"))
        self.assertIn("guessed feeling", out)
        self.assertIn("already exists", self.fails("emotion", "label", "--data", train, "--out", self.p("s.jsonl")))
        rows = E.read_samples(self.p("s.jsonl"))
        self.assertTrue({"joy", "tired", "anger", "amusement"} <= {r["label"] for r in rows})
        typed = iter(["sadness", "", "q"])
        out = self.run_cli("emotion", "review", "--samples", self.p("s.jsonl"), "--count", "5", typed=typed)
        self.assertIn("Stopped. 2 answered", out)
        self.assertEqual(sum(1 for r in E.read_samples(self.p("s.jsonl")) if r["fixed"]), 2)
        self.assertIn("corrected by you", self.run_cli("emotion", "report", "--samples", self.p("s.jsonl")))
        out = self.run_cli("emotion", "train", "--samples", self.p("s.jsonl"), "--out", self.p("model.json"), "--min-count", "1")
        self.assertIn("Wrote", out)
        self.assertIn("NOT accuracy", out)
        self.assertIn("corrected 2 messages", out)
        out = self.run_cli("emotion", "tag", "--model", self.p("model.json"), "pagod", "na", "ako")
        self.assertIn("tired", out)

    def test_emotion_train_needs_enough_samples_and_label_needs_real_data(self):
        E.write_samples(self.p("small.jsonl"), E.make_samples(["happy"], E.Tagger()))
        self.assertIn("too few", self.fails("emotion", "train", "--samples", self.p("small.jsonl"), "--out", self.p("m.json")))
        write(self.p("empty.jsonl"), '{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}\n')
        write(self.p("nothing.jsonl"), '{"text": "plain"}\n')
        self.assertIn("No messages of yours", self.fails("emotion", "label", "--data", self.p("nothing.jsonl"), "--out", self.p("x.jsonl")))
        self.assertIn("Cannot read the emotion model", self.fails("emotion", "tag", "--model", self.p("missing.json"), "hi"))

    def test_tag_memories_is_a_plan_until_apply_and_only_touches_neutral_ones(self):
        mem = self.p("mem.json")
        self.run_cli("memory", "--file", mem, "add", "I was so happy yay at the party")
        self.run_cli("memory", "--file", mem, "add", "bus leaves at nine")
        self.run_cli("memory", "--file", mem, "add", "so sad about the news", "--emotion", "stress")
        before = read(mem)
        out = self.run_cli("emotion", "tag-memories", "--memory", mem)
        self.assertIn("1 memory still tagged neutral", out)
        self.assertEqual(read(mem), before)
        out = self.run_cli("emotion", "tag-memories", "--memory", mem, "--apply")
        self.assertIn("Applied", out)
        st = M.MemoryStore(mem)
        self.assertEqual([st.memories[i]["emotion"] for i in ("m0001", "m0002", "m0003")], ["joy", "neutral", "stress"])
        self.assertIn("0 memories", self.run_cli("emotion", "tag-memories", "--memory", mem))

    def test_mood_show_and_reset(self):
        path = self.p("mood.json")
        self.assertIn("no mood file yet", self.run_cli("mood", "--file", path, "show"))
        m = E.Mood(path)
        m.update(E.Tagger().tag("galit bwisit"), m.clock())
        m.save()
        self.assertIn("recent: anger", self.run_cli("mood", "--file", path, "show"))
        self.run_cli("mood", "--file", path, "reset")
        self.assertEqual(json.loads(read(path))["valence"], 0.1)
        write(path, "{broken")
        self.assertIn("Cannot read the mood file", self.fails("mood", "--file", path, "show"))
        self.assertEqual(read(path), "{broken")

    def test_the_real_chat_command_with_mood_model_and_card(self):
        corpus = self.p("c.txt")
        write(corpus, "the cat sat on the mat. maria loves mango and the dog. hello how are you today fine thanks. name speaking style. " * 80)
        for argv in (["tokenizer", "--data", corpus, "--vocab", "300", "--out", self.p("tok.json")],
                     ["prepare", "--kind", "pretrain", "--tokenizer", self.p("tok.json"), "--data", corpus, "--out", self.p("d")],
                     ["train", "--run", self.p("run"), "--data-dir", self.p("d"), "--size", "tiny", "--steps", "2", "--backend", "numpy"]):
            self.run_cli(*argv)
        self.run_cli("person", "--file", self.p("card.md"), "init", "--name", "Jaymar")
        self.run_cli("emotion", "label", "--data", self.chats(), "--out", self.p("s.jsonl"))
        self.run_cli("emotion", "train", "--samples", self.p("s.jsonl"), "--out", self.p("model.json"), "--min-count", "1")
        typed = iter(["pagod na pagod ako, antok", "pagod na naman, antok", "/mood", "/mood tag galit ako", "/quit"])
        out = self.run_cli("chat", "--ckpt", self.p("run", "ckpt.npz"), "--tokenizer", self.p("tok.json"), "--max-new", "4", "--save-dir", self.root,
                           "--mood", self.p("mood.json"), "--emotion-model", self.p("model.json"), "--person", self.p("card.md"), typed=typed)
        self.assertIn("Mood file:", out)
        self.assertIn("Person card:", out)
        self.assertIn("feeling now: tired", out)
        self.assertIn("Feeling: tired", out)
        self.assertIn("anger", out)
        self.assertEqual(json.loads(read(self.p("mood.json")))["turns"], 2)

    def test_bad_stage5_files_stop_the_chat_before_the_model_loads_and_are_not_touched(self):
        base = ["chat", "--ckpt", self.p("none.npz"), "--tokenizer", self.p("none.json")]
        write(self.p("mood.json"), "{broken")
        self.assertIn("Cannot read the mood file", self.fails(*base, "--mood", self.p("mood.json")))
        self.assertEqual(read(self.p("mood.json")), "{broken")
        self.assertIn("Cannot read the Person card", self.fails(*base, "--person", self.p("nope.md")))
        self.assertIn("Cannot read the emotion model", self.fails(*base, "--mood", self.p("fresh.json"), "--emotion-model", self.p("nope.json")))
        self.assertIn("only work together with --mood", self.fails(*base, "--emotion-model", self.p("nope.json")))
        self.assertFalse(os.path.exists(self.p("fresh.json")))


if __name__ == "__main__":
    unittest.main()
