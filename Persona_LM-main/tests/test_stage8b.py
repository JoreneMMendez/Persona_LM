"""Stage 8b tests: the trained side of personas (owner's examples, training data that carries the exact same persona block as the chat, an attached
fine-tuned version) and the identity regression check WITH a mask on. Tiny real checkpoints come from the Stage 6 fixtures; a fake model is used where
the answer must be known. No real keyboard, no network."""
import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import chatrepl, cli
from persona_lm import data as D
from persona_lm import identity as I
from persona_lm import lineage as L
from persona_lm import personas as PS
from persona_lm.bpe import END, PERSONA, ROLE_ID, SYSTEM, USER, Tokenizer

from tests.test_stage6 import FakeGen, Fixtures, read, write, typed, sha, setUpModule as _up, tearDownModule as _down


def setUpModule():
    _up()


def tearDownModule():
    _down()


MENTOR = "You speak slowly and kindly and ask good questions."
CTX = 1024          # the fake models have no window of their own; the tiny real ones have 64, too small for any mask


class P8(Fixtures):
    def setUp(self):
        super().setUp()
        self.c = self.core_with_tokenizer()
        self.tk = Tokenizer.load(self.p("tok.json"))
        self.ps = PS.Personas(self.c, clock=self.clock)
        self.ps.add("mentor", MENTOR)

    def kinds(self):
        return [e["kind"] for e in L.Lineage(self.p("core", "lineage.jsonl")).entries()]

    def reload_ps(self):
        return PS.Personas(self.reload(), clock=self.clock)

    def lines(self, path):
        return [json.loads(x) for x in read(path).splitlines()]

    def baseline(self, reply="I am Luna and I love honesty with Jaymar"):
        """Version v1 recorded WITHOUT a mask and promoted; returns the fake model used."""
        write(self.c.probes_path(), json.dumps(I.make_probes(self.c, None, max_new=8)))
        self.c.add_version("v1", self.ck("A"))
        gen = FakeGen(self.tk, reply)
        I.record_answers(self.c, "v1", gen, self.tk, CTX)
        self.c.promote("v1")
        return gen


class RoleInData(P8):
    def test_persona_role_in_data_matches_the_chat_block(self):
        self.ps.use("mentor")
        s = chatrepl.ChatSession(self.tk, CTX, FakeGen(self.tk), personas=self.ps)
        block_ids = s._persona_block()
        text = self.ps.prompt_text()
        ids, mask = D.encode_chat(self.tk, [("persona", text, False), ("user", "hi", False), ("assistant", "hello")])
        self.assertEqual(list(ids[:len(block_ids)]), block_ids)                  # identical tokens in training and in the chat
        self.assertEqual(int(mask[:len(block_ids)].sum()), 0)                    # no loss on the mask text
        self.assertEqual(ROLE_ID["persona"], PERSONA)

    def test_data_loader_keeps_persona_messages_and_never_trains_on_them(self):
        rec = {"messages": [{"role": "persona", "content": "x"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
        self.assertEqual([r for r, _, t in D.as_messages(rec, with_train=True) if t], ["assistant"])
        self.assertIsNone(D.as_messages({"messages": [{"role": "persona", "content": "x"}, {"role": "user", "content": "a"}]}))


class Examples(P8):
    def test_add_and_reload(self):
        self.assertIn("Example 1", self.ps.add_example("mentor", "I am tired", "Rest first. What weighs on you most?"))
        self.assertEqual(self.reload_ps().get("mentor")[1]["examples"], [["I am tired", "Rest first. What weighs on you most?"]])
        self.assertIn("[1 example(s)]", self.reload_ps().describe())

    def test_rules(self):
        for u, r in (("", "x"), ("x", "  "), ("x" * 301, "y"), ("a", "b <|end|>"), ("a |> b", "c")):
            with self.assertRaises(PS.PersonaProblem, msg=(u, r)):
                self.ps.add_example("mentor", u, r)
        self.ps.add_example("mentor", "a", "b")
        with self.assertRaises(PS.PersonaProblem):
            self.ps.add_example("mentor", "a", "b")                              # duplicate
        for i in range(PS.MAX_EXAMPLES - 1):
            self.ps.add_example("mentor", "q%d" % i, "r%d" % i)
        with self.assertRaises(PS.PersonaProblem):
            self.ps.add_example("mentor", "one", "too many")
        with self.assertRaises(PS.PersonaProblem):
            self.ps.add_example("ghost", "a", "b")

    def test_logged_without_the_text(self):
        self.ps.add_example("mentor", "my secret question about Maria", "my secret answer")
        raw = read(self.p("core", "lineage.jsonl"))
        self.assertNotIn("secret", raw)
        self.assertEqual(self.kinds()[-1], "persona-edited")
        self.assertTrue(all(ok for ok, _ in self.reload().verify()))

    def test_damaged_example_in_the_file_is_refused(self):
        d = json.loads(read(self.p("core", PS.FILE_NAME)))
        d["personas"]["mentor"]["examples"] = [["ok", "forged <|end|>"]]
        write(self.p("core", PS.FILE_NAME), json.dumps(d))
        with self.assertRaises(PS.PersonaProblem):
            self.reload_ps()


class MakeData(P8):
    def setUp(self):
        super().setUp()
        self.ps.add_example("mentor", "I failed my test", "That hurts. What will you try next?")
        self.ps.add_example("mentor", "hi", "Hello. Take your time.")
        self.out = self.p("data", "mentor.jsonl")

    def test_lines_have_no_loss_except_the_reply_and_the_chat_block(self):
        self.assertEqual(self.ps.make_data("mentor", self.out, self.tk, CTX), 2)
        recs = self.lines(self.out)
        self.assertEqual(len(recs), 2)
        m = recs[0]["messages"]
        self.assertEqual([x["role"] for x in m], ["system", "persona", "user", "assistant"])
        self.assertEqual([x.get("train", True) for x in m], [False, False, False, True])
        self.assertEqual(m[1]["content"], self.ps.get("mentor") and PS.block_text("Luna", "mentor", MENTOR))
        self.assertIn("still Luna underneath", m[1]["content"])
        ids, mask = D.encode_chat(self.tk, D.as_messages(recs[0], with_train=True))
        self.assertEqual(int(mask.sum()), len(self.tk.encode("That hurts. What will you try next?", allow_special=False)) + 1)     # reply + its <|end|>
        self.assertIn(PERSONA, list(ids))
        self.assertEqual(len(list(D.chat_docs([self.out], self.tk))), 2)           # the real data loader reads the file

    def test_same_tokens_as_the_chat_prompt(self):
        self.ps.make_data("mentor", self.out, self.tk, CTX)
        self.ps.use("mentor")
        gen = FakeGen(self.tk, "ok")
        sysline, _ = I.system_line(self.c.sections, lambda t: len(self.tk.encode(t, allow_special=False)), max(((CTX - 80) // 2) - 4, 8))    # the same limit make_data uses
        s = chatrepl.ChatSession(self.tk, CTX, gen, system=sysline, personas=self.ps)
        s.handle("I failed my test")
        trained, _ = D.encode_chat(self.tk, D.as_messages(self.lines(self.out)[0], with_train=True))
        prompt = gen.calls[-1][0]
        trained = [int(x) for x in trained]
        upto = trained.index(ROLE_ID["assistant"], trained.index(USER))      # everything before the reply
        self.assertEqual(prompt[:-1], trained[:upto])                              # the prompt is exactly the training context

    def test_never_overwrites_and_needs_examples(self):
        self.ps.make_data("mentor", self.out, self.tk, CTX)
        before = read(self.out)
        with self.assertRaises(PS.PersonaProblem):
            self.ps.make_data("mentor", self.out, self.tk, CTX)
        self.assertEqual(before, read(self.out))
        self.ps.add("empty", "Be brief.")
        with self.assertRaises(PS.PersonaProblem):
            self.ps.make_data("empty", self.p("x.jsonl"), self.tk, CTX)
        self.assertFalse(os.path.exists(self.p("x.jsonl")))

    def test_window_too_small_refused_and_logged_with_a_hash(self):
        with self.assertRaises(PS.PersonaProblem):
            self.ps.make_data("mentor", self.p("tiny.jsonl"), self.tk, 40)
        self.assertFalse(os.path.exists(self.p("tiny.jsonl")))
        self.ps.make_data("mentor", self.out, self.tk, CTX)
        e = L.Lineage(self.p("core", "lineage.jsonl")).entries()[-1]
        self.assertEqual((e["kind"], e["data"]["file_sha256"], e["data"]["examples"]), ("persona-data", sha(self.out), 2))
        self.assertNotIn("That hurts", read(self.p("core", "lineage.jsonl")))


class Attach(P8):
    def test_attach_detach_stale_and_log(self):
        self.c.add_version("v1", self.ck("A"))
        self.c.add_version("v2", self.ck("B"), parent="v1")
        self.assertIn("attached", self.ps.attach("mentor", "v2"))
        self.assertIn("Already", self.ps.attach("mentor", "v2"))
        self.assertEqual(self.reload_ps().version_for("mentor"), "v2")
        self.assertFalse(self.ps.stale("mentor"))
        self.assertIn("OLD text", self.ps.edit("mentor", "You are stern."))
        self.assertTrue(self.reload_ps().stale("mentor"))
        self.assertIn("STALE", self.reload_ps().describe())
        self.ps.attach("mentor", "v2")                                           # re-attaching after retraining clears it
        self.assertFalse(self.reload_ps().stale("mentor"))
        self.assertIn("Detached", self.ps.detach("mentor"))
        self.assertIsNone(self.reload_ps().version_for("mentor"))
        self.assertIn("no version", self.ps.detach("mentor"))
        self.assertEqual(self.kinds().count("persona-edited"), 4)

    def test_unknown_or_rejected_versions_are_refused(self):
        self.c.add_version("v1", self.ck("A"))
        with self.assertRaises(PS.PersonaProblem):
            self.ps.attach("mentor", "nope")
        c = self.reload()
        c.versions["v1"]["status"] = "rejected"
        c.save()
        ps = PS.Personas(c, clock=self.clock)
        with self.assertRaises(PS.PersonaProblem):
            ps.attach("mentor", "v1")
        c.versions["v1"]["status"] = "candidate"
        c.save()
        ps.attach("mentor", "v1")
        c.versions["v1"]["status"] = "rejected"                                  # rejected AFTER attaching: refused at use time, not replaced silently
        c.save()
        with self.assertRaises(PS.PersonaProblem) as cm:
            PS.Personas(c, clock=self.clock).version_for("mentor")
        self.assertIn("REJECTED", str(cm.exception))

    def test_removing_the_persona_forgets_the_attachment(self):
        self.c.add_version("v1", self.ck("A"))
        self.ps.attach("mentor", "v1")
        self.ps.remove("mentor")
        self.assertEqual(self.reload_ps().names(), [])


class MaskCheck(P8):
    def check(self, reply, **kw):
        gen = FakeGen(self.tk, reply)
        res = PS.check_mask(self.c, self.ps, "mentor", gen, self.tk, CTX, out=lambda *_: None, **kw)
        return res, gen

    def test_facts_kept_passes_and_the_mask_really_was_on(self):
        base = self.baseline()
        self.assertTrue(all(PERSONA not in ids for ids, _ in base.calls))        # the baseline had no mask
        res, gen = self.check("I am Luna and I love honesty with Jaymar, said slowly")
        self.assertTrue(res["judged"] and res["passed"], res)
        self.assertTrue(all(PERSONA in ids for ids, _ in gen.calls))              # every probe was asked WITH the mask
        text = PS.describe_check("mentor", res)
        self.assertIn("OK: the mask does not hide who it is", text)
        self.assertIn("not judged", text)

    def test_mask_that_hides_the_facts_is_flagged(self):
        self.baseline()
        res, _ = self.check("banana banana banana")
        self.assertTrue(res["judged"] and not res["passed"])
        self.assertEqual(res["recall"], 0.0)
        self.assertIn("TOO LOW", PS.describe_check("mentor", res))

    def test_style_changes_alone_do_not_fail_it(self):
        self.baseline()
        res, _ = self.check("Hmm... " + "I am Luna and I love honesty with Jaymar" + " ... take your time, friend")
        self.assertTrue(res["passed"])
        self.assertLess(res["categories"]["similarity"], 1.0)

    def test_changes_nothing_but_adds_one_log_line_with_numbers(self):
        self.baseline()
        files = {n: read(self.p("core", n)) for n in ("self.md", "probes.json", PS.FILE_NAME)}
        answers = read(self.c.answers_path("v1"))
        sections, current, status = dict(self.reload().sections), self.reload().current, self.reload().versions["v1"]["status"]
        n = len(self.kinds())
        self.check("I am Luna and I love honesty with Jaymar")
        self.assertEqual(files, {k: read(self.p("core", k)) for k in files})
        self.assertEqual(answers, read(self.c.answers_path("v1")))                # the baseline is not overwritten
        r = self.reload()
        self.assertEqual((dict(r.sections), r.current, r.versions["v1"]["status"]), (sections, current, status))
        self.assertEqual((len(self.kinds()), self.kinds()[-1]), (n + 1, "persona-checked"))
        self.assertTrue(all(ok for ok, _ in r.verify()))
        e = L.Lineage(self.p("core", "lineage.jsonl")).entries()[-1]
        self.assertEqual(e["data"]["persona"], "mentor")
        self.assertNotIn(MENTOR, read(self.p("core", "lineage.jsonl")))

    def test_it_does_not_switch_the_mask_on(self):
        self.baseline()
        self.check("I am Luna and I love honesty with Jaymar")
        self.assertIsNone(self.reload_ps().active)

    def test_refusals(self):
        with self.assertRaises(I.IdentityProblem):                                # nothing promoted yet
            self.check("x")
        self.baseline()
        write(self.c.probes_path(), json.dumps(dict(I.load_probes(self.c), made="changed")))
        with self.assertRaises(I.IdentityProblem) as cm:
            self.check("x")
        self.assertIn("probes.json changed", str(cm.exception))
        with self.assertRaises(PS.PersonaProblem):
            PS.check_mask(self.c, self.ps, "ghost", FakeGen(self.tk), self.tk, CTX)

    def test_nothing_to_judge_is_said_plainly(self):
        self.baseline()
        doc = I.load_probes(self.c)
        for p in doc["probes"]:
            p["expect"] = []
        write(self.c.probes_path(), json.dumps(doc))
        I.record_answers(self.c, "v1", FakeGen(self.tk, "hi"), self.tk, CTX)
        res, _ = self.check("hi")
        self.assertFalse(res["judged"] or res["passed"])
        self.assertIn("nothing can be judged", PS.describe_check("mentor", res))

    def test_a_mask_that_does_not_fit_is_said_plainly_not_reported_as_lost_facts(self):
        self.baseline()
        self.ps.add("huge", " ".join(["mentor hello thanks"] * 30)[:PS.MAX_TEXT_CHARS])
        with self.assertRaises(PS.PersonaProblem) as cm:
            PS.check_mask(self.c, self.ps, "huge", FakeGen(self.tk, ""), self.tk, CTX, out=lambda *_: None)
        self.assertIn("gave no answer to any probe", str(cm.exception))
        n = len(self.kinds())
        self.assertEqual(len(self.kinds()), n)                                    # nothing was logged for a test that could not run

    def test_some_silent_probes_count_as_lost_and_are_noted(self):
        self.baseline()
        gen = FakeGen(self.tk, "I am Luna and I love honesty with Jaymar")
        answers = iter(range(100))
        orig = gen.__call__
        def flaky(ids, **kw):
            gen.calls.append((list(ids), kw))
            return [] if next(answers) < 2 else self.tk.encode(gen.reply, allow_special=False)
        res = PS.check_mask(self.c, self.ps, "mentor", flaky, self.tk, CTX, out=lambda *_: None)
        self.assertTrue(any("got no answer" in n for n in res["notes"]))


class ChatPicksTheVersion(P8):
    def chat(self, *extra, fail=False):
        buf = io.StringIO()
        argv = ["chat", "--core", self.p("core"), "--max-new", "4"] + list(extra)
        with mock.patch("builtins.input", typed("/quit")), contextlib.redirect_stdout(buf):
            if fail:
                with self.assertRaises(SystemExit) as cm:
                    cli.main(argv)
                return str(cm.exception) + buf.getvalue()
            cli.main(argv)
        return buf.getvalue()

    def setUp(self):
        super().setUp()
        self.baseline()
        self.c.add_version("v2", self.ck("B"), parent="v1")
        I.record_answers(self.c, "v2", FakeGen(self.tk), self.tk, CTX)

    def test_attached_version_is_used_while_the_mask_is_on(self):
        self.ps.attach("mentor", "v2")
        self.assertIn("Version: v1.", self.chat())                                # mask off: the current version
        self.ps.use("mentor")
        out = self.chat()
        self.assertIn("Version: v2", out)
        self.assertIn("[an OLDER version, not the current one]", out)
        self.assertIn("Persona (a mask): mentor", out)
        self.assertIn("fine-tuned for this mask", out)
        self.assertIn("Version: v1.", self.chat("--version", "v1"))               # the owner's own choice wins

    def test_stale_warning_and_rejected_refusal(self):
        self.ps.attach("mentor", "v2")
        self.ps.use("mentor")
        self.ps.edit("mentor", "You are stern.")
        self.assertIn("WARNING: the mask text changed", self.chat())
        c = self.reload()
        c.versions["v2"]["status"] = "rejected"
        c.save()
        self.assertIn("REJECTED", self.chat(fail=True))

    def test_no_attachment_means_the_current_version(self):
        self.ps.use("mentor")
        out = self.chat()
        self.assertIn("Version: v1.", out)
        self.assertNotIn("fine-tuned for this mask", out)


class CommandLine(P8):
    def cli(self, *argv, fail=False):
        buf = io.StringIO()
        base = ["persona", "--dir", self.p("core")]
        with contextlib.redirect_stdout(buf):
            if fail:
                with self.assertRaises(SystemExit) as cm:
                    cli.main(base + list(argv))
                return str(cm.exception) + buf.getvalue()
            cli.main(base + list(argv))
        return buf.getvalue()

    def test_the_whole_trained_side_with_real_tiny_models(self):
        before = (json.dumps(self.reload().sections, sort_keys=True), read(self.p("core", "self.md")))
        self.assertIn("Example 1", self.cli("example", "mentor", "I failed my test", "That hurts. What will you try next?"))
        self.assertIn("Example 2", self.cli("example", "mentor", "hi", "Hello. Take your time."))
        out = self.cli("make-data", "mentor", "--out", self.p("m.jsonl"), "--ctx", "2048")
        self.assertIn("Wrote 2 example(s)", out)
        self.assertIn("not a small adapter", out)
        self.assertIn("already exists", self.cli("make-data", "mentor", "--out", self.p("m.jsonl"), "--ctx", "2048", fail=True))
        self.assertIn("no version to test", self.cli("check", "mentor", fail=True))
        self.c.add_version("v1", self.ck("A"))
        write(self.c.probes_path(), json.dumps(I.make_probes(self.c, None, max_new=6)))
        gen, tk, ctx = I.load_runner(self.c, "v1")
        I.record_answers(self.c, "v1", gen, tk, ctx)
        self.c.promote("v1")
        out = self.cli("check", "mentor", fail=True)                              # the tiny practice models have a 64-token window: no mask can fit
        self.assertIn("with the mask 'mentor' on", out)
        self.assertIn("gave no answer to any probe", out)
        self.assertNotIn("persona-checked", self.kinds())
        self.assertIn("Version 'v1' is attached", self.cli("attach", "mentor", "v1").replace("\"", "'"))
        self.assertIn("[trained version: v1]", self.cli("list"))
        self.assertIn("Detached", self.cli("detach", "mentor"))
        r = self.reload()
        self.assertEqual(before, (json.dumps(r.sections, sort_keys=True), read(self.p("core", "self.md"))))
        self.assertTrue(all(ok for ok, _ in r.verify()))

    def test_errors_are_plain(self):
        self.assertIn("no persona called", self.cli("example", "ghost", "a", "b", fail=True))
        self.assertIn("no persona called", self.cli("check", "ghost", fail=True))
        self.assertIn("no version called", self.cli("attach", "mentor", "nope", fail=True))


if __name__ == "__main__":
    unittest.main()
