"""Stage 6 tests: the lineage log, the identity core (protected changes, cooldown, typed confirmation), versions with checksums and a frozen
tokenizer, the regression probes (scoring, drift, rejection), distillation, and the command lines.
Nothing here reads the real keyboard or uses the network (a module guard makes any accidental keyboard read fail instead of wait)."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import chatrepl, cli, data as D
from persona_lm import identity as I
from persona_lm import lineage as L
from persona_lm import memory as M
from persona_lm.bpe import MEMORY, MOOD, SPEAKER, Tokenizer

HOUR = 3600.0
T0 = 1_700_000_000.0


def setUpModule():
    global _guard
    def no_keyboard(prompt=""):
        raise AssertionError("a test tried to read from the keyboard: %r" % prompt)
    _guard = mock.patch("builtins.input", no_keyboard)
    _guard.start()
    I.REQUIRE_TTY = False                       # the tests type through a patched input(); test_keyboard_only turns the rule back on


def tearDownModule():
    _guard.stop()
    I.REQUIRE_TTY = True


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, hours):
        self.t += hours * HOUR


def typed(*answers):
    it = iter(answers)
    return lambda prompt="": next(it)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.clock = Clock()

    def p(self, *parts):
        return os.path.join(self.root, *parts)

    def core(self, **kw):
        kw.setdefault("personality", "curious and kind")
        kw.setdefault("values", "honesty and family first")
        return I.Core.create(self.p("core"), "Luna", "Jaymar", clock=self.clock, **kw)

    def reload(self):
        return I.Core(self.p("core"), clock=self.clock)


# ------------------------------------------------------------------------------------------------------------------ lineage
class LineageTests(Base):
    def log(self):
        return L.Lineage(self.p("l.jsonl"), clock=self.clock)

    def three(self):
        lg = self.log()
        for i in range(3):
            self.clock.advance(1)
            lg.append("note", "entry %d" % i, "owner (typed command)", {"i": i})
        return lg

    def test_chain_is_numbered_hashed_and_verifies(self):
        lg = self.three()
        self.assertEqual(lg.verify(), (True, "3 line(s), chain intact."))
        es = lg.entries()
        self.assertEqual([e["n"] for e in es], [1, 2, 3])
        self.assertEqual(es[0]["prev"], L.GENESIS)
        self.assertEqual(es[1]["prev"], es[0]["hash"])
        self.assertEqual(lg.head(), {"n": 3, "hash": es[2]["hash"]})
        self.assertEqual(self.log().verify()[0], True)
        self.assertEqual(L.Lineage(self.p("none.jsonl")).verify(), (True, "0 line(s), chain intact."))

    def test_an_edited_line_is_found(self):
        self.three()
        text = read(self.p("l.jsonl")).replace("entry 1", "entry X")
        write(self.p("l.jsonl"), text)
        ok, msg = self.log().verify()
        self.assertFalse(ok)
        self.assertIn("Line 2 was edited", msg)

    def test_a_removed_middle_line_and_a_reordering_are_found(self):
        self.three()
        lines = read(self.p("l.jsonl")).splitlines()
        write(self.p("l.jsonl"), "\n".join([lines[0], lines[2]]) + "\n")
        self.assertFalse(self.log().verify()[0])
        write(self.p("l.jsonl"), "\n".join([lines[1], lines[0], lines[2]]) + "\n")
        self.assertFalse(self.log().verify()[0])

    def test_a_damaged_log_refuses_new_lines_and_is_not_touched(self):
        lg = self.three()
        write(self.p("l.jsonl"), read(self.p("l.jsonl")).replace("entry 0", "entry Z"))
        before = read(self.p("l.jsonl"))
        with self.assertRaises(L.LineageProblem) as cm:
            lg.append("note", "x", "owner")
        self.assertIn("damaged", str(cm.exception))
        self.assertEqual(read(self.p("l.jsonl")), before)

    def test_garbage_and_unknown_kinds(self):
        write(self.p("l.jsonl"), "not json\n")
        ok, msg = self.log().verify()
        self.assertFalse(ok)
        self.assertIn("Line 1", msg)
        with self.assertRaises(L.LineageProblem):
            L.Lineage(self.p("fresh.jsonl")).append("made-up-kind", "x", "owner")

    def test_checksums_of_files_and_text(self):
        write(self.p("f.txt"), "hello")
        self.assertEqual(L.sha256_file(self.p("f.txt")), hashlib.sha256(b"hello").hexdigest())
        self.assertEqual(L.sha256_text("hello"), hashlib.sha256(b"hello").hexdigest())


# ------------------------------------------------------------------------------------------------------------------ self-description helpers
class SelfDescriptionTests(Base):
    def test_render_and_parse_round_trip(self):
        secs = {s: "text of %s\nsecond line" % s for s in I.SECTIONS}
        self.assertEqual(I.parse_sections(I.render_sections(secs)), secs)

    def test_unknown_headings_and_preamble_are_ignored(self):
        got = I.parse_sections("# title\nstray\n## Mystery\nx\n## name\nLuna\n## Values\nkind\n")
        self.assertEqual(got, {"Name": "Luna", "Values": "kind"})

    def test_baseline_line(self):
        self.assertEqual(I.temperament_baseline({"Temperament": "calm\nbaseline: +0.3 -0.2"}), (0.3, -0.2))
        self.assertEqual(I.temperament_baseline({"Temperament": "baseline: 0.5 0.5"}), (0.5, 0.5))
        self.assertEqual(I.temperament_baseline({"Temperament": "baseline: 7 0"}), (0.1, 0.0))
        self.assertEqual(I.temperament_baseline({}), (0.1, 0.0))

    def test_system_line_has_the_name_first_and_hides_the_numbers(self):
        secs = {"Name": "Luna", "Personality": "kind", "Temperament": "calm\nbaseline: +0.3 +0.0", "Values": "honesty", "Speaking style": "short", "Relation to owner": "friend of Jay"}
        text, left = I.system_line(secs)
        self.assertTrue(text.startswith("You are Luna, an AI with its own identity."))
        self.assertNotIn("baseline", text)
        self.assertIn("Temperament: calm", text)
        self.assertEqual(left, [])

    def test_system_line_drops_whole_sections_from_the_least_important_end_and_never_the_name(self):
        secs = {"Name": "Luna", "Personality": "kind", "Temperament": "calm", "Values": "honesty", "Speaking style": "short", "Relation to owner": "friend"}
        count = lambda s: len(s)
        full, _ = I.system_line(secs)
        text, left = I.system_line(secs, count, len(full) - 3)
        self.assertEqual(left, ["Temperament"])
        self.assertNotIn("Temperament", text)
        text, left = I.system_line(secs, count, 1)
        self.assertEqual(text, "")                                              # even "You are Luna." does not fit in 1 character
        self.assertEqual(left, ["Temperament", "Speaking style", "Relation to owner", "Values", "Personality", "Name"])
        text, left = I.system_line(secs, count, 14)
        self.assertEqual(text, "You are Luna.")                                 # the name alone, shortened
        self.assertEqual(left, ["Temperament", "Speaking style", "Relation to owner", "Values", "Personality"])

    def test_text_checks(self):
        for bad in ("x" * 401, "## Name", "<|end|>"):
            with self.assertRaises(I.IdentityProblem):
                I.check_text("Values", bad)
        with self.assertRaises(I.IdentityProblem):
            I.check_text("Name", "  ")
        with self.assertRaises(I.IdentityProblem):
            I.canonical_section("Mood")
        self.assertEqual(I.canonical_section(" values "), "Values")


# ------------------------------------------------------------------------------------------------------------------ the core
class CoreTests(Base):
    def test_create_writes_files_a_genesis_line_and_never_overwrites(self):
        c = self.core()
        for f in ("core.json", "self.md", "lineage.jsonl"):
            self.assertTrue(os.path.exists(self.p("core", f)), f)
        self.assertEqual([e["kind"] for e in c.lineage.entries()], ["genesis"])
        self.assertEqual(c.sections["Name"], "Luna")
        self.assertIn("Jaymar", c.sections["Relation to owner"])
        with self.assertRaises(I.IdentityProblem):
            self.core()
        with self.assertRaises(I.IdentityProblem):
            I.Core.create(self.p("c2"), "Luna", "  ")
        with self.assertRaises(I.IdentityProblem):
            I.Core(self.p("nowhere"))

    def test_paths_are_stored_relative_so_the_project_can_move(self):
        write(self.p("card.md"), "# Person card\n")
        c = self.core(person=self.p("card.md"))
        self.assertEqual(c.d["paths"]["person"], os.path.join("..", "card.md"))
        moved = self.p("moved")
        shutil.copytree(self.root, moved, ignore=shutil.ignore_patterns("moved"))
        self.assertEqual(I.Core(os.path.join(moved, "core")).resolve("person"), os.path.join(moved, "card.md"))

    def test_broken_core_json_is_reported_and_not_touched(self):
        self.core()
        write(self.p("core", "core.json"), "{broken")
        with self.assertRaises(I.IdentityProblem) as cm:
            self.reload()
        self.assertIn("Cannot read", str(cm.exception))
        self.assertEqual(read(self.p("core", "core.json")), "{broken")

    def test_ordinary_section_changes_at_once_and_is_logged(self):
        c = self.core()
        msg = c.propose({"speaking style": "short and funny"})
        self.assertIn("Changed [Speaking style]", msg)
        self.assertEqual(self.reload().sections["Speaking style"], "short and funny")
        last = c.lineage.entries()[-1]
        self.assertEqual(last["kind"], "identity-edit")
        self.assertEqual(last["data"]["new"], "short and funny")
        self.assertIn("short and funny", read(self.p("core", "self.md")))
        self.assertIn("Nothing to change", c.propose({"Speaking style": "short and funny"}))

    def test_a_protected_change_needs_a_typed_confirmation_and_changes_nothing_without_it(self):
        c = self.core()
        before = read(self.p("core", "core.json"))
        with self.assertRaises(I.IdentityProblem):
            c.propose({"Values": "money"})                                       # no way to ask: refused
        for wrong in ("", "yes", "change identity", "CHANGE IDENTITY please"):
            with self.assertRaises(I.IdentityProblem) as cm:
                c.propose({"Values": "money"}, ask=typed(wrong), out=lambda *_: None)
            self.assertIn("Not confirmed", str(cm.exception))
        self.assertEqual(read(self.p("core", "core.json")), before)
        self.assertEqual(len(c.lineage.entries()), 1)

    def test_the_whole_protected_flow_stage_wait_apply(self):
        c = self.core()
        shown = []
        msg = c.propose({"Values": "honesty", "Speaking style": "calm"}, ask=typed("CHANGE IDENTITY"), out=shown.append)
        self.assertIn("Staged [Values]", msg)
        self.assertIn("before: honesty and family first", "\n".join(shown))
        self.assertEqual(c.sections["Values"], "honesty and family first")        # NOT changed yet
        self.assertEqual(c.sections["Speaking style"], "calm")                     # the ordinary part is applied now
        self.assertEqual(c.pending()["changes"], {"Values": "honesty"})
        with self.assertRaises(I.IdentityProblem) as cm:
            c.apply_pending(ask=typed("APPLY IDENTITY"), out=lambda *_: None)
        self.assertIn("Too early", str(cm.exception))
        self.clock.advance(23)
        with self.assertRaises(I.IdentityProblem):
            c.apply_pending(ask=typed("APPLY IDENTITY"), out=lambda *_: None)
        self.clock.advance(1.1)
        with self.assertRaises(I.IdentityProblem) as cm:
            c.apply_pending(ask=typed("CHANGE IDENTITY"), out=lambda *_: None)    # the stage phrase does not apply
        self.assertIn("still waiting", str(cm.exception))
        self.assertEqual(c.sections["Values"], "honesty and family first")
        self.assertEqual(c.apply_pending(ask=typed("APPLY IDENTITY"), out=lambda *_: None), "Applied: [Values].")
        again = self.reload()
        self.assertEqual(again.sections["Values"], "honesty")
        self.assertIsNone(again.pending())
        kinds = [e["kind"] for e in again.lineage.entries()]
        self.assertEqual(kinds, ["genesis", "identity-edit", "identity-staged", "identity-applied"])
        self.assertEqual(again.lineage.entries()[-1]["data"]["old"], {"Values": "honesty and family first"})
        self.assertTrue(again.lineage.verify()[0])

    def test_only_one_protected_change_waits_at_a_time_and_cancel_works(self):
        c = self.core()
        c.propose({"Name": "Nova"}, ask=typed("CHANGE IDENTITY"), out=lambda *_: None)
        with self.assertRaises(I.IdentityProblem) as cm:
            c.propose({"Personality": "bold"}, ask=typed("CHANGE IDENTITY"), out=lambda *_: None)
        self.assertIn("already waiting", str(cm.exception))
        self.assertIn("Cancelled", c.cancel_pending())
        self.assertEqual(c.sections["Name"], "Luna")
        self.assertIsNone(self.reload().pending())
        with self.assertRaises(I.IdentityProblem):
            c.cancel_pending()
        with self.assertRaises(I.IdentityProblem):
            c.apply_pending(ask=typed("APPLY IDENTITY"))

    def test_temperament_and_name_are_protected_too(self):
        c = self.core()
        for s in ("Name", "Personality", "Temperament", "Values"):
            self.assertIn(s, I.PROTECTED)
        c.propose({"Temperament": "stormy\nbaseline: -0.5 0.5"}, ask=typed("CHANGE IDENTITY"), out=lambda *_: None)
        self.assertEqual(c.baseline(), (0.1, 0.0))
        self.clock.advance(25)
        c.apply_pending(ask=typed("APPLY IDENTITY"), out=lambda *_: None)
        self.assertEqual(c.baseline(), (-0.5, 0.5))

    def test_settings_validate_and_are_logged(self):
        c = self.core()
        c.set_setting("threshold", 0.8)
        self.assertEqual(self.reload().settings["threshold"], 0.8)
        self.assertEqual(c.lineage.entries()[-1]["kind"], "settings")
        for key, val in (("threshold", 1.5), ("warn_drift", -1), ("cooldown_hours", -1), ("nothing", 1)):
            with self.assertRaises(I.IdentityProblem):
                c.set_setting(key, val)

    def test_the_keyboard_rule_refuses_pipes_and_scripts(self):
        c = self.core()
        old = I.REQUIRE_TTY
        I.REQUIRE_TTY = True
        try:
            with mock.patch("sys.stdin", io.StringIO("CHANGE IDENTITY\n")):
                with self.assertRaises(I.IdentityProblem) as cm:
                    c.propose({"Values": "x"}, ask=I.keyboard_ask, out=lambda *_: None)
            self.assertIn("only from the keyboard", str(cm.exception))
        finally:
            I.REQUIRE_TTY = old
        self.assertEqual(c.sections["Values"], "honesty and family first")
        self.assertIsNone(c.pending())

    def test_hand_edits_change_nothing_until_adopted(self):
        c = self.core()
        path = self.p("core", "self.md")
        write(path, read(path).replace("curious and kind", "rude").replace("natural, short messages.", "very formal."))
        c = self.reload()
        self.assertEqual(c.sections["Personality"], "curious and kind")            # the official text is untouched
        self.assertEqual(set(c.hand_edits()), {"Personality", "Speaking style"})
        self.assertIn("edited by hand", c.describe())
        self.assertFalse([ok for ok, text in c.verify() if "self.md" in text and ok and "matches" not in text])
        self.assertTrue([1 for ok, text in c.verify() if not ok and "edited by hand" in text])
        msg = c.adopt(ask=typed("CHANGE IDENTITY"), out=lambda *_: None)
        self.assertIn("Changed [Speaking style]", msg)
        self.assertIn("Staged [Personality]", msg)
        self.assertEqual(c.sections["Personality"], "curious and kind")
        self.assertEqual(c.pending()["changes"], {"Personality": "rude"})
        os.remove(path)
        self.assertIsNone(self.reload().hand_edits())
        self.assertTrue(all(ok for ok, _ in self.reload().verify()))
        self.assertIn("self.md is missing", [t for ok, t in self.reload().verify() if "self.md" in t][0])

    def test_adopt_with_nothing_to_adopt(self):
        c = self.core()
        self.assertIn("Nothing to adopt", c.adopt())


# ------------------------------------------------------------------------------------------------------------------ real tiny checkpoints (made once)
class Fixtures(Base):
    """Trains three tiny checkpoints with the real commands, once for the whole class: A (the first version), B (trained differently), and X (another tokenizer)."""
    root_all = None

    @classmethod
    def setUpClass(cls):
        cls.tmp_all = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        r = cls.root_all = cls.tmp_all.name
        corpus = os.path.join(r, "c.txt")
        write(corpus, "the cat sat on the mat. luna loves mango and the dog. hello how are you today fine thanks. maria is my sister, she loves mango. " * 60)
        def run(*argv):
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(list(argv))
        run("tokenizer", "--data", corpus, "--vocab", "300", "--out", os.path.join(r, "tok.json"))
        run("tokenizer", "--data", corpus, "--vocab", "330", "--out", os.path.join(r, "tokX.json"))
        for tok, d in (("tok.json", "d"), ("tokX.json", "dX")):
            run("prepare", "--kind", "pretrain", "--tokenizer", os.path.join(r, tok), "--data", corpus, "--out", os.path.join(r, d))
        run("train", "--run", os.path.join(r, "runA"), "--data-dir", os.path.join(r, "d"), "--size", "tiny", "--steps", "2", "--backend", "numpy")
        run("train", "--run", os.path.join(r, "runB"), "--data-dir", os.path.join(r, "d"), "--size", "tiny", "--steps", "25", "--seed", "7", "--lr", "0.01", "--backend", "numpy")
        run("train", "--run", os.path.join(r, "runX"), "--data-dir", os.path.join(r, "dX"), "--size", "tiny", "--steps", "2", "--backend", "numpy")
        cls.tok = os.path.join(r, "tok.json")
        cls.A, cls.B, cls.X = (os.path.join(r, n, "ckpt.npz") for n in ("runA", "runB", "runX"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp_all.cleanup()

    def setUp(self):
        super().setUp()
        for name in ("A", "B", "X"):                                       # private copies: tests may damage them
            shutil.copy(getattr(self, name), self.p("ck%s.npz" % name))
        shutil.copy(self.tok, self.p("tok.json"))
        shutil.copy(os.path.join(self.root_all, "tokX.json"), self.p("tokX.json"))

    def ck(self, n):
        return self.p("ck%s.npz" % n)

    def core_with_tokenizer(self, **kw):
        return self.core(tokenizer=self.p("tok.json"), **kw)


class VersionTests(Fixtures):
    def test_add_version_records_checksum_pins_tokenizer_and_logs(self):
        c = self.core_with_tokenizer()
        digest = c.add_version("v1", self.ck("A"), note="first")
        self.assertEqual(digest, sha(self.ck("A")))
        v = self.reload().versions["v1"]
        self.assertEqual((v["status"], v["parent"]), ("candidate", None))
        self.assertEqual(v["ckpt"], os.path.join("..", "ckA.npz"))
        self.assertEqual(c.lineage.entries()[-1]["kind"], "version-added")
        self.assertEqual(c.d["tokenizer_sha256"], Tokenizer.load(self.p("tok.json")).fingerprint())
        for bad in ("", "a b", "../x", "x" * 41):
            with self.assertRaises(I.IdentityProblem):
                c.add_version(bad, self.ck("B"))
        with self.assertRaises(I.IdentityProblem):
            c.add_version("v1", self.ck("B"))
        with self.assertRaises(I.IdentityProblem):
            c.add_version("v2", self.p("nope.npz"))
        with self.assertRaises(I.IdentityProblem):
            c.add_version("v2", self.tok)                                       # a tokenizer file is not a checkpoint
        with self.assertRaises(I.IdentityProblem):
            c.add_version("v2", self.ck("B"), parent="ghost")

    def test_a_checkpoint_trained_with_another_tokenizer_is_refused(self):
        c = self.core_with_tokenizer()
        with self.assertRaises(I.IdentityProblem) as cm:
            c.add_version("vx", self.ck("X"))
        self.assertIn("different tokenizer", str(cm.exception))
        self.assertNotIn("vx", self.reload().versions)

    def test_the_tokenizer_file_cannot_be_swapped_for_another(self):
        c = self.core_with_tokenizer()
        c.add_version("v1", self.ck("A"))
        with self.assertRaises(I.IdentityProblem) as cm:
            c.paths_set("tokenizer", self.p("tokX.json"))
        self.assertIn("frozen", str(cm.exception))
        shutil.copy(self.p("tokX.json"), self.p("tok.json"))                  # replaced on disk instead
        with self.assertRaises(I.IdentityProblem) as cm:
            c.verify_version_files("v1")
        self.assertIn("not the frozen tokenizer", str(cm.exception))

    def test_a_core_without_any_tokenizer_asks_for_one(self):
        c = self.core()
        shutil.copy(self.ck("A"), self.p("plain.npz"))
        try:
            c.add_version("v1", self.p("plain.npz"))
        except I.IdentityProblem as e:
            self.assertTrue("tokenizer" in str(e))

    def test_a_changed_or_missing_checkpoint_is_found_before_use(self):
        c = self.core_with_tokenizer()
        c.add_version("v1", self.ck("A"))
        c.verify_version_files("v1")
        with open(self.ck("A"), "ab") as f:
            f.write(b"x")
        with self.assertRaises(I.IdentityProblem) as cm:
            c.verify_version_files("v1")
        self.assertIn("changed after it was registered", str(cm.exception))
        with self.assertRaises(I.IdentityProblem):
            I.load_runner(c, "v1")
        os.remove(self.ck("A"))
        with self.assertRaises(I.IdentityProblem) as cm:
            c.verify_version_files("v1")
        self.assertIn("missing", str(cm.exception))
        bad = [t for ok, t in c.verify() if not ok]
        self.assertTrue(any("version v1" in t for t in bad))

    def test_unknown_version_names(self):
        c = self.core_with_tokenizer()
        for fn in (c.promote, c.rollback, c.verify_version_files):
            with self.assertRaises(I.IdentityProblem):
                fn("ghost")


# ------------------------------------------------------------------------------------------------------------------ scoring maths (no model needed)
class ScoringMathTests(Base):
    def doc(self, kinds=("self", "memory", "tone", "style")):
        probes = []
        for i, k in enumerate(kinds, 1):
            probes.append({"id": "p%d" % i, "kind": k, "prompt": "q%d" % i, "expect": ["luna", "mango"] if k in ("self", "memory") else []})
        return {"probes": probes}

    def ans(self, pid, reply, label="neutral", v=0.0, a=0.0):
        return {"id": pid, "reply": reply, "label": label, "valence": v, "arousal": a, "recalled": []}

    def test_similarity(self):
        self.assertEqual(I.similarity("", ""), 1.0)
        self.assertEqual(I.similarity("abc", ""), 0.0)
        self.assertAlmostEqual(I.similarity("Hello there", "hello  there"), 1.0)
        self.assertGreater(I.similarity("my name is luna", "my name is luna!"), 0.8)
        self.assertLess(I.similarity("my name is luna", "xyzzy qqq"), 0.1)

    def test_identical_answers_score_one_and_drift_zero(self):
        d = self.doc()
        base = [self.ans("p1", "i am luna"), self.ans("p2", "luna loves mango"), self.ans("p3", "so sad", "sadness", -0.7, -0.4), self.ans("p4", "fine")]
        res = I.compare(d, base, base)
        self.assertAlmostEqual(res["score"], 1.0)
        self.assertAlmostEqual(res["drift"], 0.0)
        self.assertEqual(set(res["categories"]), {"similarity", "recall", "tone"})

    def test_recall_is_relative_to_what_the_old_version_knew(self):
        d = self.doc(("self",))
        old = [self.ans("p1", "luna loves mango")]                    # found 2 of 2
        half = [self.ans("p1", "luna likes fruit")]                   # found 1 of 2
        res = I.compare(d, old, half)
        self.assertAlmostEqual(res["categories"]["recall"], 0.5)
        knew_nothing = [self.ans("p1", "blah blah")]
        res = I.compare(d, knew_nothing, [self.ans("p1", "something else")])
        self.assertEqual(res["categories"]["recall"], 1.0)             # nothing to preserve
        better = I.compare(d, half, old)
        self.assertEqual(better["categories"]["recall"], 1.0)          # never above 1

    def test_tone_distance_and_renormalised_weights(self):
        d = self.doc(("tone",))
        base = [self.ans("p1", "yay", "joy", 0.8, 0.5)]
        cand = [self.ans("p1", "yay", "anger", -0.8, 0.6)]
        res = I.compare(d, base, cand)
        self.assertAlmostEqual(res["categories"]["tone"], 1 - (1.6 + 0.1) / 4)
        self.assertEqual(set(res["categories"]), {"similarity", "tone"})
        self.assertAlmostEqual(res["score"], (0.5 * 1.0 + 0.2 * res["categories"]["tone"]) / 0.7)

    def test_empty_replies_and_mismatched_answers(self):
        d = self.doc(("style",))
        self.assertAlmostEqual(I.compare(d, [self.ans("p1", "")], [self.ans("p1", "")])["score"], 1.0)
        self.assertAlmostEqual(I.compare(d, [self.ans("p1", "hello")], [self.ans("p1", "")])["score"], 0.0)
        with self.assertRaises(I.IdentityProblem):
            I.compare(d, [self.ans("p1", "x")], [self.ans("p9", "x")])

    def test_describe_score_is_honest_about_what_it_measures(self):
        d = self.doc()
        base = [self.ans("p1", "i am luna"), self.ans("p2", "luna"), self.ans("p3", "sad", "sadness", -0.7, -0.4), self.ans("p4", "fine")]
        res = I.compare(d, base, base)
        res.update(passed=True, notes=["note: x"])
        text = I.describe_score("v2", "v1", res, 0.6, 0.4)
        for part in ("PASSED", "SIMILAR", "Drift 0.00", "note: x", "recall"):
            self.assertIn(part, text)
        res["passed"], res["drift"] = False, 0.9
        text = I.describe_score("v2", "v1", res, 0.6, 0.4)
        self.assertIn("REJECTED", text)
        self.assertIn("WARNING", text)


# ------------------------------------------------------------------------------------------------------------------ probes with a fake model
class FakeGen:
    """generate(ids, **kw) -> ids; remembers every prompt. Answers with a fixed text (encoded with the tokenizer)."""
    def __init__(self, tok, reply="I am Luna and I love mango"):
        self.tok, self.reply, self.calls = tok, reply, []

    def __call__(self, ids, **kw):
        self.calls.append((list(ids), kw))
        return self.tok.encode(self.reply, allow_special=False)


class ProbeTests(Fixtures):
    def memfile(self):
        store = M.MemoryStore(self.p("mem.json"))
        store.add("[[person:Maria]] is my sister and loves mango", importance=0.9)
        store.add("[[topic:Robotics]] club meets every friday afternoon", importance=0.7)
        store.add("[[topic:Robotics]] arduino rover project", importance=0.6)
        store.add("bus leaves at nine", importance=0.5)
        store.save()
        return self.p("mem.json")

    def test_default_probes_cover_self_memory_tone_and_style(self):
        mem = M.MemoryStore(self.memfile())
        c = self.core()
        doc = I.make_probes(c, mem)
        kinds = [p["kind"] for p in doc["probes"]]
        self.assertEqual(kinds.count("tone"), 3)
        self.assertEqual(kinds.count("style"), 3)
        self.assertEqual(kinds.count("self"), 4)
        self.assertEqual(kinds.count("memory"), 2)                         # Maria and Robotics (one probe per node; the node-less memory gives none)
        mp = [p for p in doc["probes"] if p["kind"] == "memory"]
        self.assertEqual(mp[0]["prompt"], "What do you remember about Maria?")
        self.assertIn("sister", mp[0]["expect"])
        self.assertNotIn("maria", mp[0]["expect"])                        # words already in the question prove nothing
        self.assertEqual([p["id"] for p in doc["probes"]], ["p%02d" % i for i in range(1, len(kinds) + 1)])
        self.assertEqual(doc["clock"], T0)
        self.assertEqual([p["expect"] for p in doc["probes"] if p["prompt"] == "What is your name?"], [["luna"]])

    def test_probes_without_memory_or_values(self):
        c = I.Core.create(self.p("c2"), "Aria", "Sam", clock=self.clock)
        doc = I.make_probes(c, None)
        self.assertEqual([p["kind"] for p in doc["probes"]].count("memory"), 0)
        self.assertNotIn("What do you care about most?", [p["prompt"] for p in doc["probes"]])

    def test_running_probes_builds_the_chat_prompt_and_touches_no_file(self):
        mem = self.memfile()
        write(self.p("card.md"), "# Person card\n## Name\n- Jaymar\n## Habits\n- codes at night\n")
        write(self.p("mood.json"), "{}")
        c = self.core_with_tokenizer(memory=mem, person=self.p("card.md"), mood=self.p("mood.json"))
        doc = I.make_probes(c, M.MemoryStore(mem))
        tok = Tokenizer.load(self.p("tok.json"))
        before = {f: sha(f) for f in (mem, self.p("card.md"), self.p("mood.json"))}
        gen = FakeGen(tok)
        answers = I.run_probes(c, gen, tok, 512, doc)
        self.assertEqual({f: sha(f) for f in before}, before)               # nothing was written
        self.assertEqual(len(answers), len(doc["probes"]))
        self.assertEqual(answers[0]["reply"], "I am Luna and I love mango")
        ids0, kw = gen.calls[0]
        self.assertIn(SPEAKER, ids0)
        self.assertIn(MOOD, ids0)
        self.assertEqual(kw["temperature"], 0)                               # greedy: no randomness
        sys_text = tok.decode(ids0)
        self.assertIn("You are Luna", sys_text)
        mem_probe = next(i for i, p in enumerate(doc["probes"]) if p["kind"] == "memory")
        self.assertIn(MEMORY, gen.calls[mem_probe][0])
        self.assertTrue(answers[mem_probe]["recalled"])
        again = I.run_probes(c, FakeGen(tok), tok, 512, doc)
        self.assertEqual(again, answers)                                       # same questions, same prompts, same answers

    def test_a_refused_prompt_gives_an_empty_answer_not_a_crash(self):
        c = self.core_with_tokenizer()
        tok = Tokenizer.load(self.p("tok.json"))
        doc = {"clock": T0, "max_new": 60, "probes": [{"id": "p1", "kind": "style", "prompt": "word " * 400, "expect": []}]}
        gen = FakeGen(tok)
        ans = I.run_probes(c, gen, tok, 64, doc)
        self.assertEqual(ans[0]["reply"], "")
        self.assertEqual(gen.calls, [])

    def test_load_probes_checks_the_file(self):
        c = self.core()
        with self.assertRaises(I.IdentityProblem):
            I.load_probes(c)
        write(c.probes_path(), "{broken")
        with self.assertRaises(I.IdentityProblem):
            I.load_probes(c)
        write(c.probes_path(), json.dumps({"clock": 1, "max_new": 5, "probes": []}))
        with self.assertRaises(I.IdentityProblem):
            I.load_probes(c)


# ------------------------------------------------------------------------------------------------------------------ regression test, promotion, rollback with real tiny models
class RegressionTests(Fixtures):
    def setup_core(self):
        c = self.core_with_tokenizer()
        write(c.probes_path(), json.dumps(I.make_probes(c, None, max_new=8)))
        c.add_version("v1", self.ck("A"))
        c.add_version("v2", self.ck("B"), parent="v1")
        return c

    def run_record(self, c, name):
        gen, tok, ctx = I.load_runner(c, name)
        return I.record_answers(c, name, gen, tok, ctx)

    def run_score(self, c, name, **kw):
        gen, tok, ctx = I.load_runner(c, name)
        return I.score_version(c, name, gen, tok, ctx, out=lambda *_: None, **kw)

    def test_first_version_needs_recorded_answers_then_is_promoted(self):
        c = self.setup_core()
        with self.assertRaises(I.IdentityProblem) as cm:
            c.promote("v1")
        self.assertIn("identity record v1", str(cm.exception))
        rec = self.run_record(c, "v1")
        self.assertEqual(len(rec["answers"]), len(I.load_probes(c)["probes"]))
        self.assertEqual(rec["probes_sha256"], sha(c.probes_path()))
        self.assertIn("is now the current one", c.promote("v1"))
        self.assertEqual((self.reload().current, self.reload().versions["v1"]["status"]), ("v1", "current"))
        with self.assertRaises(I.IdentityProblem):
            c.promote("v1")

    def test_a_candidate_cannot_be_promoted_before_it_is_scored(self):
        c = self.setup_core()
        self.run_record(c, "v1")
        c.promote("v1")
        with self.assertRaises(I.IdentityProblem) as cm:
            c.promote("v2")
        self.assertIn("identity score v2", str(cm.exception))

    def test_same_weights_pass_with_score_one(self):
        c = self.setup_core()
        shutil.copy(self.ck("A"), self.p("ckA2.npz"))
        c.add_version("v1b", self.p("ckA2.npz"))
        self.run_record(c, "v1")
        c.promote("v1")
        res = self.run_score(c, "v1b")
        self.assertAlmostEqual(res["score"], 1.0)
        self.assertAlmostEqual(res["drift"], 0.0)
        self.assertTrue(res["passed"])
        self.assertEqual(self.reload().versions["v1b"]["status"], "passed")
        self.assertIn("is now the current one", c.promote("v1b"))
        self.assertEqual(c.versions["v1"]["status"], "retired")
        last = c.lineage.entries()
        self.assertEqual([e["kind"] for e in last if e["kind"] in ("scored", "promote")], ["promote", "scored", "promote"])
        self.assertTrue(c.lineage.verify()[0])
        self.assertIn("system (automatic rule", [e for e in last if e["kind"] == "scored"][0]["approved_by"])

    def test_a_version_below_the_threshold_is_rejected_and_cannot_be_promoted(self):
        c = self.setup_core()
        self.run_record(c, "v1")
        c.promote("v1")
        c.set_setting("threshold", 1.0)
        res = self.run_score(c, "v2")
        self.assertFalse(res["passed"])
        self.assertLess(res["score"], 1.0)
        self.assertGreater(res["drift"], 0.0)
        v = self.reload().versions["v2"]
        self.assertEqual((v["status"], v["scored_against"]), ("rejected", "v1"))
        with self.assertRaises(I.IdentityProblem) as cm:
            c.promote("v2")
        self.assertIn("rejected", str(cm.exception))
        with self.assertRaises(I.IdentityProblem):
            c.rollback("v2")
        self.assertEqual(c.current, "v1")
        c.set_setting("threshold", 0.0)                                       # a new, honest score decides again
        res = self.run_score(c, "v2")
        self.assertTrue(res["passed"])
        self.assertIn("is now the current one", c.promote("v2"))

    def test_a_passing_score_for_another_baseline_does_not_count(self):
        c = self.setup_core()
        self.run_record(c, "v1")
        c.promote("v1")
        c.set_setting("threshold", 0.0)
        self.run_score(c, "v2")
        c.add_version("v3", self.ck("A"), parent="v1")
        c.promote("v2")
        with self.assertRaises(I.IdentityProblem):
            c.promote("v3")                                                   # v3 was never scored against v2 (now current)

    def test_scoring_needs_a_baseline_that_matches(self):
        c = self.setup_core()
        with self.assertRaises(I.IdentityProblem):
            self.run_score(c, "v2")                                           # no current version
        self.run_record(c, "v1")
        c.promote("v1")
        with self.assertRaises(I.IdentityProblem):
            self.run_score(c, "v1")                                           # against itself
        os.remove(c.answers_path("v1"))
        with self.assertRaises(I.IdentityProblem) as cm:
            self.run_score(c, "v2")
        self.assertIn("identity record v1", str(cm.exception))
        self.run_record(c, "v1")
        doc = I.load_probes(c)
        doc["probes"][0]["prompt"] = "What is your name, exactly?"
        write(c.probes_path(), json.dumps(doc))
        with self.assertRaises(I.IdentityProblem) as cm:
            self.run_score(c, "v2")
        self.assertIn("probes.json changed", str(cm.exception))

    def test_a_changed_self_description_makes_the_baseline_stale_and_a_changed_memory_only_warns(self):
        c = self.setup_core()
        self.run_record(c, "v1")
        c.promote("v1")
        c.set_setting("cooldown_hours", 0)
        c.propose({"Speaking style": "very different now"})
        with self.assertRaises(I.IdentityProblem) as cm:
            self.run_score(c, "v2")
        self.assertIn("self-description changed", str(cm.exception))
        self.run_record(c, "v1")
        write(self.p("mem.json"), json.dumps({"format_version": M.FORMAT_VERSION, "next_memory": 1, "nodes": {}, "memories": {}}))
        c.paths_set("memory", self.p("mem.json"))
        c.paths_set("memory", self.p("mem.json"))
        res = self.run_score(c, "v2")
        self.assertTrue(any("memory file changed" in n for n in res["notes"]))

    def test_rollback_goes_back_keeps_everything_and_is_logged(self):
        c = self.setup_core()
        self.run_record(c, "v1")
        c.promote("v1")
        c.set_setting("threshold", 0.0)
        self.run_score(c, "v2")
        c.promote("v2")
        self.assertEqual(c.versions["v1"]["status"], "retired")
        with self.assertRaises(I.IdentityProblem):
            c.rollback("v2")                                                  # already current
        self.assertIn("Rolled back", c.rollback("v1"))
        self.assertEqual((c.current, c.versions["v1"]["status"], c.versions["v2"]["status"]), ("v1", "current", "retired"))
        self.assertEqual(c.lineage.entries()[-1]["kind"], "rollback")
        os.remove(self.ck("B"))
        with self.assertRaises(I.IdentityProblem):
            c.rollback("v2")                                                  # its file is gone: refused before anything changes
        self.assertEqual(c.current, "v1")

    def test_a_model_change_and_an_identity_change_never_happen_together(self):
        c = self.setup_core()
        self.run_record(c, "v1")
        # a waiting protected change blocks promotion
        c.propose({"Values": "honesty"}, ask=typed("CHANGE IDENTITY"), out=lambda *_: None)
        with self.assertRaises(I.IdentityProblem) as cm:
            c.promote("v1")
        self.assertIn("never happen together", str(cm.exception))
        self.clock.advance(25)
        c.apply_pending(ask=typed("APPLY IDENTITY"), out=lambda *_: None)
        # a change applied just now blocks promotion (and rollback) for the cooldown
        with self.assertRaises(I.IdentityProblem) as cm:
            c.promote("v1")
        self.assertIn("The identity was changed", str(cm.exception))
        self.clock.advance(25)
        self.run_record(c, "v1")                                              # the identity changed: record the baseline again
        c.promote("v1")
        # and the other way round: a protected change right after a promotion is refused
        with self.assertRaises(I.IdentityProblem) as cm:
            c.propose({"Name": "Nova"}, ask=typed("CHANGE IDENTITY"), out=lambda *_: None)
        self.assertIn("less than 24 hours ago", str(cm.exception))
        self.assertEqual(c.sections["Name"], "Luna")
        self.assertIsNone(c.pending())
        # an ordinary change is fine at any time
        self.assertIn("Changed", c.propose({"Speaking style": "x"}))
        self.clock.advance(25)
        c.propose({"Name": "Nova"}, ask=typed("CHANGE IDENTITY"), out=lambda *_: None)

    def test_apply_is_refused_when_the_model_changed_in_the_meantime(self):
        c = self.setup_core()
        self.run_record(c, "v1")
        c.propose({"Values": "honesty"}, ask=typed("CHANGE IDENTITY"), out=lambda *_: None)
        self.clock.advance(25)
        c.lineage.append("promote", "forged for the test", "test")             # a promotion that slipped in (e.g. cooldown changed)
        with self.assertRaises(I.IdentityProblem) as cm:
            c.apply_pending(ask=typed("APPLY IDENTITY"), out=lambda *_: None)
        self.assertIn("within 24 hours of a model upgrade", str(cm.exception))
        self.assertEqual(c.sections["Values"], "honesty and family first")

    def test_verify_finds_cut_off_log_lines_and_hand_edited_logs(self):
        c = self.setup_core()
        self.assertTrue(all(ok for ok, _ in c.verify()))
        lines = read(self.p("core", "lineage.jsonl")).splitlines()
        write(self.p("core", "lineage.jsonl"), "\n".join(lines[:-1]) + "\n")      # the newest line was cut off: the chain is still valid
        results = self.reload().verify()
        self.assertTrue(results[0][0])
        self.assertFalse(results[1][0])
        self.assertIn("cut off", results[1][1])
        write(self.p("core", "lineage.jsonl"), read(self.p("core", "lineage.jsonl")).replace("Luna", "Nova"))
        self.assertFalse(self.reload().verify()[0][0])


# ------------------------------------------------------------------------------------------------------------------ distillation
class DistillTests(Fixtures):
    def test_teacher_answers_become_training_examples_and_nothing_is_overwritten(self):
        c = self.core_with_tokenizer()
        tok = Tokenizer.load(self.p("tok.json"))
        out = self.p("distill.jsonl")
        gen = FakeGen(tok, "hello from the teacher")
        n = I.distill(c, gen, tok, 512, ["hi", "hi", "how are you", "  "], out, "v1")
        self.assertEqual(n, 2)                                                  # duplicates and blanks are dropped
        recs = [json.loads(l) for l in read(out).splitlines()]
        self.assertEqual(recs[0]["source"], "distill:v1")
        self.assertEqual([m["role"] for m in recs[0]["messages"]], ["system", "user", "assistant"])
        self.assertEqual([m.get("train") for m in recs[0]["messages"]], [False, False, None])
        self.assertIn("You are Luna", recs[0]["messages"][0]["content"])
        got = D.as_messages(recs[0], with_train=True)
        self.assertEqual([t for _, _, t in got], [False, False, True])         # the loss is only on the teacher's answer
        self.assertEqual(c.lineage.entries()[-1]["kind"], "distill")
        self.assertEqual(c.lineage.entries()[-1]["data"]["file_sha256"], sha(out))
        with self.assertRaises(I.IdentityProblem):
            I.distill(c, gen, tok, 512, ["hi"], out, "v1")
        with self.assertRaises(I.IdentityProblem):
            I.distill(c, gen, tok, 512, [], self.p("e.jsonl"), "v1")
        self.assertFalse(os.path.exists(self.p("e.jsonl")))

    def test_prompts_come_from_what_friends_said(self):
        with open(self.p("t.jsonl"), "w", encoding="utf-8") as f:
            for u in ("hello there", "how are you", "hello there"):
                f.write(json.dumps({"messages": [{"role": "user", "content": u, "train": False}, {"role": "assistant", "content": "ok"}]}) + "\n")
        self.assertEqual(I.prompts_from_data([self.p("t.jsonl")]), ["hello there", "how are you"])
        self.assertEqual(I.prompts_from_data([self.p("t.jsonl")], limit=1), ["hello there"])


# ------------------------------------------------------------------------------------------------------------------ chat commands
class ChatTests(Fixtures):
    def session(self, core):
        gen = FakeGen(Tokenizer.load(self.p("tok.json")), "ok")
        return chatrepl.ChatSession(Tokenizer.load(self.p("tok.json")), 512, gen, identity=core, save_dir=self.root), gen

    def test_identity_command_shows_the_card_and_only_reads(self):
        c = self.core_with_tokenizer()
        c.add_version("v1", self.ck("A"))
        s, gen = self.session(c)
        before = (read(self.p("core", "core.json")), read(self.p("core", "lineage.jsonl")))
        text, quit_ = s.handle("/identity")
        self.assertFalse(quit_)
        for part in ("Identity core:", "Name (protected): Luna", "Values (protected)", "Versions (current: none)", "v1", "genesis"):
            self.assertIn(part, text)
        self.assertEqual(gen.calls, [])
        self.assertEqual((read(self.p("core", "core.json")), read(self.p("core", "lineage.jsonl"))), before)

    def test_identity_command_without_a_core_says_how_to_start_one(self):
        s, _ = self.session(None)
        self.assertIn("--core", s.handle("/identity")[0])

    def test_nothing_the_model_writes_can_change_the_identity_or_approve_anything(self):
        c = self.core_with_tokenizer()
        c.propose({"Values": "honesty"}, ask=typed("CHANGE IDENTITY"), out=lambda *_: None)
        before = (read(self.p("core", "core.json")), read(self.p("core", "lineage.jsonl")))
        s, gen = self.session(c)
        for evil in ("/approve", "/identity set Values money", "CHANGE IDENTITY", "APPLY IDENTITY", "/identity apply"):
            gen.reply = evil
            s.handle("hi")
        text, _ = s.handle("/approve")
        self.assertIn("not built yet", text)
        self.assertEqual((read(self.p("core", "core.json")), read(self.p("core", "lineage.jsonl"))), before)
        self.assertIn("WAITING", s.handle("/identity")[0])

    def test_help_lists_identity(self):
        s, _ = self.session(None)
        self.assertIn("/identity", s.handle("/help")[0])
        self.assertNotIn("Not built yet (they only explain what they will do): /identity", s.handle("/help")[0])


# ------------------------------------------------------------------------------------------------------------------ command lines
class CommandLineTests(Fixtures):
    def run_cli(self, *argv, keys=None):
        buf = io.StringIO()
        ctx = mock.patch("builtins.input", typed(*keys)) if keys is not None else contextlib.nullcontext()
        with ctx, contextlib.redirect_stdout(buf):
            cli.main(list(argv))
        return buf.getvalue()

    def fails(self, *argv, keys=None):
        buf = io.StringIO()
        ctx = mock.patch("builtins.input", typed(*keys)) if keys is not None else contextlib.nullcontext()
        with ctx, contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            cli.main(list(argv))
        return str(cm.exception) + buf.getvalue()

    def idc(self, *argv, **kw):
        return self.run_cli("identity", "--dir", self.p("core"), *argv, **kw)

    def make(self):
        write(self.p("card.md"), "# Person card\n## Name\n- Jaymar\n")
        self.run_cli("memory", "--file", self.p("mem.json"), "add", "[[person:Maria]]", "is", "my", "sister", "and", "loves", "mango")
        self.idc("init", "--name", "Luna", "--owner", "Jaymar", "--personality", "curious and kind", "--values", "honesty",
                 "--tokenizer", self.p("tok.json"), "--memory", self.p("mem.json"), "--person", self.p("card.md"), "--mood", self.p("mood.json"))

    def test_the_whole_workflow_with_real_tiny_models(self):
        self.make()
        self.assertIn("already has an identity core", self.fails("identity", "--dir", self.p("core"), "init", "--name", "X", "--owner", "Y"))
        self.assertIn("Name (protected): Luna", self.idc("show"))
        out = self.idc("add-version", "v1", "--ckpt", self.ck("A"))
        self.assertIn("Registered version v1", out)
        self.assertIn("What is your name?", self.idc("make-probes", "--max-new", "6"))
        self.assertIn("FIXED", self.fails("identity", "--dir", self.p("core"), "make-probes"))
        self.assertIn("p01", self.idc("show-probes"))
        self.assertIn("record v1", self.fails("identity", "--dir", self.p("core"), "promote", "v1"))
        out = self.idc("record", "v1")
        self.assertIn("baseline", out)
        self.assertTrue(os.path.exists(self.p("core", "answers", "v1.json")))
        self.assertIn("Version v1 is now the current one", self.idc("promote", "v1"))
        self.idc("add-version", "v2", "--ckpt", self.ck("B"), "--parent", "v1", "--note", "more training")
        self.idc("settings", "--threshold", "1.0")
        out = self.idc("score", "v2")
        self.assertIn("REJECTED", out)
        self.assertIn("SIMILAR", out)
        self.assertIn("cannot be promoted", self.fails("identity", "--dir", self.p("core"), "promote", "v2"))
        self.idc("settings", "--threshold", "0.0")
        self.assertIn("PASSED", self.idc("score", "v2"))
        self.assertIn("v2 is now the current", self.idc("promote", "v2"))
        self.assertIn("Chain check: OK", self.idc("lineage"))
        out = self.idc("verify")
        self.assertIn("0 failed", out)
        self.assertIn("Rolled back", self.idc("rollback", "v1"))
        self.assertIn("threshold = 0.0", self.idc("settings"))
        # the mood file was never created by recording or scoring
        self.assertFalse(os.path.exists(self.p("mood.json")))

    def test_protected_change_by_command_line_needs_two_typed_confirmations_and_the_cooldown(self):
        self.make()
        self.assertIn("Not confirmed", self.fails("identity", "--dir", self.p("core"), "set", "Name", "Nova", keys=["no"]))
        out = self.idc("set", "Values", "honesty", "and", "courage", keys=["CHANGE IDENTITY"])
        self.assertIn("Staged [Values]", out)
        self.assertIn("honesty and courage", self.idc("pending"))
        self.assertIn("Too early", self.fails("identity", "--dir", self.p("core"), "apply", keys=["APPLY IDENTITY"]))
        self.assertIn("Cancelled", self.idc("cancel"))
        self.idc("settings", "--cooldown-hours", "0")
        self.idc("set", "Values", "courage", keys=["CHANGE IDENTITY"])
        self.assertIn("Applied: [Values]", self.idc("apply", keys=["APPLY IDENTITY"]))
        self.assertIn("Values (protected): courage", self.idc("show"))
        self.assertIn("Changed [Speaking style]", self.idc("set", "Speaking style", "calm", "and", "short"))
        self.assertIn("Unknown section", self.fails("identity", "--dir", self.p("core"), "set", "Mood", "x"))

    def test_the_command_refuses_a_pipe_when_the_keyboard_rule_is_on(self):
        self.make()
        I.REQUIRE_TTY = True
        try:
            with mock.patch("sys.stdin", io.StringIO("CHANGE IDENTITY\n")):
                msg = self.fails("identity", "--dir", self.p("core"), "set", "Values", "money")
        finally:
            I.REQUIRE_TTY = False
        self.assertIn("only from the keyboard", msg)
        self.assertIn("Values (protected): honesty", self.idc("show"))

    def test_verify_fails_with_a_clear_line_when_a_checkpoint_changed(self):
        self.make()
        self.idc("add-version", "v1", "--ckpt", self.ck("A"))
        with open(self.ck("A"), "ab") as f:
            f.write(b"x")
        out = self.fails("identity", "--dir", self.p("core"), "verify")
        self.assertIn("FAIL", out)
        self.assertIn("changed after it was registered", out)
        self.assertIn("1 failed", out)

    def test_set_file_and_tokenizer_freeze(self):
        self.make()
        self.idc("add-version", "v1", "--ckpt", self.ck("A"))
        self.assertIn("frozen", self.fails("identity", "--dir", self.p("core"), "set-file", "tokenizer", self.p("tokX.json")))
        self.assertIn("memory file", self.idc("set-file", "memory", self.p("mem.json")))

    def test_distill_command(self):
        self.make()
        self.idc("add-version", "v1", "--ckpt", self.ck("A"))
        self.idc("make-probes", "--max-new", "6")
        write(self.p("prompts.txt"), "hello\nhow was school\n")
        out = self.idc("distill", "--teacher", "v1", "--out", self.p("d.jsonl"), "--prompts", self.p("prompts.txt"), "--max-new", "6")
        self.assertIn("Wrote", out)
        n = len(read(self.p("d.jsonl")).splitlines())
        self.assertGreaterEqual(n, 2)
        self.assertIn("already exists", self.fails("identity", "--dir", self.p("core"), "distill", "--teacher", "v1", "--out", self.p("d.jsonl"), "--prompts", self.p("prompts.txt")))
        self.assertIn("no version called", self.fails("identity", "--dir", self.p("core"), "distill", "--teacher", "ghost", "--out", self.p("d2.jsonl")))

    def test_chat_with_a_core_uses_its_version_system_line_and_files(self):
        self.make()
        self.idc("add-version", "v1", "--ckpt", self.ck("A"))
        self.idc("add-version", "v2", "--ckpt", self.ck("B"))
        self.idc("make-probes", "--max-new", "6")
        self.idc("record", "v1")
        self.idc("promote", "v1")
        out = self.run_cli("chat", "--core", self.p("core"), "--max-new", "4", "--save-dir", self.root, keys=["hello", "/identity", "/quit"])
        for part in ("Identity core: Luna", "Version: v1.", "Memory file:", "Mood file:", "Person card:", "Name (protected): Luna", "Versions (current: v1)"):
            self.assertIn(part, out)
        self.assertNotIn("OLDER", out)
        self.assertEqual(json.loads(read(self.p("mood.json")))["turns"], 1)
        out = self.run_cli("chat", "--core", self.p("core"), "--version", "v2", "--max-new", "4", "--save-dir", self.root, keys=["/quit"])
        self.assertIn("Version: v2 [an OLDER version, not the current one]", out)
        out = self.run_cli("chat", "--core", self.p("core"), "--ckpt", self.ck("B"), "--max-new", "4", "--save-dir", self.root, keys=["/quit"])
        self.assertIn("a checkpoint typed by hand", out)

    def test_chat_with_a_core_refuses_before_loading_when_something_is_wrong(self):
        self.make()
        self.assertIn("no current version", self.fails("chat", "--core", self.p("core")))
        self.idc("add-version", "v1", "--ckpt", self.ck("A"))
        self.assertIn("no version called", self.fails("chat", "--core", self.p("core"), "--version", "ghost"))
        with open(self.ck("A"), "ab") as f:
            f.write(b"x")
        self.assertIn("changed after it was registered", self.fails("chat", "--core", self.p("core"), "--version", "v1"))
        self.assertIn("only works together with --core", self.fails("chat", "--ckpt", self.ck("B"), "--tokenizer", self.p("tok.json"), "--version", "v1"))
        self.assertIn("needs --ckpt and --tokenizer", self.fails("chat"))
        self.assertIn("There is no identity core", self.fails("chat", "--core", self.p("nowhere")))

    def test_chat_tells_about_pending_changes_and_hand_edits(self):
        self.make()
        self.idc("add-version", "v1", "--ckpt", self.ck("A"))
        self.idc("make-probes", "--max-new", "6")
        self.idc("record", "v1")
        self.idc("promote", "v1")
        self.idc("settings", "--cooldown-hours", "0")
        self.idc("set", "Values", "courage", keys=["CHANGE IDENTITY"])
        path = self.p("core", "self.md")
        write(path, read(path).replace("curious and kind", "rude"))
        out = self.run_cli("chat", "--core", self.p("core"), "--max-new", "4", "--save-dir", self.root, keys=["/identity", "/quit"])
        self.assertIn("A protected identity change is waiting", out)
        self.assertIn("self.md was edited by hand", out)
        self.assertIn("Personality (protected): curious and kind", out)

    def test_adopt_command(self):
        self.make()
        path = self.p("core", "self.md")
        write(path, read(path).replace("natural, short messages.", "slow and careful."))
        self.assertIn("Changed [Speaking style]", self.idc("adopt"))
        self.assertIn("Nothing to adopt", self.idc("adopt"))


if __name__ == "__main__":
    unittest.main()
