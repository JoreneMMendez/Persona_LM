"""Stage 7, the four proposals the owner chose: P1 chat-probe, P2 plan-steps, P6 doctor --windows, P8 quickstart docs."""
import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from persona_lm import cli, doctor, planner, pretty, probe, selfcheck  # noqa: E402
from persona_lm.model import PRESETS  # noqa: E402
from persona_lm.train import train  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = ["hi", "hello there", "kumusta ka", "unsa imong ginabuhat", "where are you going", "what is your name"]


def pairs(replies):
    return list(zip(PROMPTS * 5, replies))


class FakeTok:
    def encode(self, text, allow_special=True):
        return [ord(c) % 50 + 10 for c in text]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class Probe(unittest.TestCase):
    def test_normalize_and_echo(self):
        self.assertEqual(probe.normalize("  Ok!!  Ra, ko... "), "ok ra ko")
        self.assertTrue(probe.is_echo("Kumusta ka?", "kumusta ka"))
        self.assertTrue(probe.is_echo("Kumusta ka?", "Kumusta ka? ok ra ko"))
        self.assertFalse(probe.is_echo("Kumusta ka?", "ok ra ko"))
        self.assertFalse(probe.is_echo("hi", ""))
        self.assertFalse(probe.is_echo("see you tomorrow", "ok"))                   # a short word inside the message is not an echo

    def test_same_stock_reply_is_a_problem(self):
        r = probe.analyse(pairs(["Kuyog ko"] * 20 + ["asa ka", "ok", "hahaha", "unsa man", "wala", "sige", "nice", "oo", "dili", "ahay"]))
        self.assertEqual(r["level"], doctor.PROBLEM)
        self.assertIn("STOCK REPLY", r["verdict"])
        self.assertEqual(r["top_count"], 20)

    def test_varied_replies_are_ok(self):
        r = probe.analyse(pairs(["reply number %d about %s" % (i, w) for i, w in enumerate("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu one two three four".split())]))
        self.assertEqual(r["level"], doctor.OK)
        self.assertEqual(r["top_count"], 1)

    def test_echo_and_empty_and_tiny(self):
        echo = probe.analyse([(p, p) for p in PROMPTS])
        self.assertEqual(echo["level"], doctor.WARN)
        self.assertEqual(probe.analyse(pairs([""] * 30))["level"], doctor.PROBLEM)
        self.assertEqual(probe.analyse([("a", "b")] * 2)["level"], doctor.WARN)
        half = probe.analyse(pairs(["x%d unique" % i for i in range(15)] + [""] * 15))
        self.assertEqual(half["empty"], 15)
        self.assertEqual(half["level"], doctor.WARN)

    def test_run_probe_uses_the_chat_format_and_seeds(self):
        seen = []

        def gen(ids, seed):
            seen.append((ids, seed))
            return [1, 2]
        out = probe.run_probe(gen, FakeTok(), ["hi", "yo"], seeds=2, system="be kind")
        self.assertEqual(len(out), 4)
        from persona_lm.bpe import ASSISTANT, END, ROLE_ID
        ids = seen[0][0]
        self.assertEqual(ids[0], ROLE_ID["system"])
        self.assertEqual(ids[-1], ASSISTANT)
        self.assertEqual(ids.count(END), 2)
        self.assertEqual([s for _, s in seen], [0, 1, 0, 1])

    def test_builtin_prompts_are_plain_and_enough(self):
        self.assertGreaterEqual(len(probe.PROMPTS), 15)
        self.assertEqual(len(set(probe.PROMPTS)), len(probe.PROMPTS))

    def test_report_hides_replies_unless_asked(self):
        ps = pairs(["Kuyog ko"] * 30)
        r = probe.analyse(ps)
        quiet = "\n".join(probe.format_report("m", r, ps, show=False))
        loud = "\n".join(probe.format_report("m", r, ps, show=True))
        self.assertNotIn("Kuyog ko", quiet)
        self.assertIn("--show", quiet)
        self.assertIn("Kuyog ko", loud)

    def test_command_runs_end_to_end_on_a_tiny_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok, cfg, data = selfcheck.make_tiny_data(tmp)
            tokp = os.path.join(tmp, "tok.json")
            tok.save(tokp)
            run = os.path.join(tmp, "run_chat")
            train(run, data, cfg, dict(selfcheck.HP), log=lambda *_: None, backup=False)
            a = cli.build_parser().parse_args(["chat-probe", "--ckpt", run, run + os.sep + "ckpt.npz", "--tokenizer", tokp, "--max-new", "5", "--seeds", "1"])
            out, lines = io.StringIO(), []
            with redirect_stdout(out):
                try:
                    probe.cmd(a, say=lines.append)
                except SystemExit as e:
                    self.assertEqual(e.code, 1)             # an untrained-ish tiny model may well be a PROBLEM; only 0 or 1 is allowed
            text = "\n".join(lines)
            self.assertEqual(text.count("messages sent: 20"), 2)      # a folder and a ckpt path both work
            self.assertIn("checkpoint", text)                         # the comparison table
            self.assertNotIn("  ->  ", text)                         # replies stay hidden

    def test_missing_tokenizer_is_a_friendly_error(self):
        a = cli.build_parser().parse_args(["chat-probe", "--ckpt", "nope", "--tokenizer", os.path.join(tempfile.gettempdir(), "no_such_tok.json")])
        with self.assertRaises(SystemExit) as cm:
            probe.cmd(a, say=lambda *_: None)
        self.assertIn("tokenizer", str(cm.exception))


class Planner(unittest.TestCase):
    S = {"kind": "chat", "train_tokens": 4_116_902, "val_tokens": 433_857, "loss_tokens": 900_000}

    def test_three_passes_matches_the_run_that_worked(self):
        p = planner.make_plan(self.S, 4096, 1, 4)
        self.assertEqual(p["per_step"], 16384)
        self.assertEqual(p["steps"], 754)                    # your 800-step run was about 3.2 passes
        self.assertAlmostEqual(planner.passes_for(800, 4_116_902, 16384), 3.18, places=1)
        self.assertEqual(p["warnings"], [])

    def test_small_data_matches_run_chat3(self):
        p = planner.make_plan(dict(self.S, train_tokens=574_600), 4096, 1, 4)
        self.assertEqual((p["steps"], p["eval_every"], p["warmup"]), (105, 15, 8))

    def test_eval_every_and_warmup_are_sane(self):
        for steps in (1, 5, 40, 105, 800, 5000, 100000):
            self.assertGreaterEqual(planner.pick_eval_every(steps), 5)
            self.assertGreaterEqual(planner.pick_warmup(steps), 1)
            self.assertLessEqual(planner.pick_warmup(steps), 100)
        self.assertEqual(planner.pick_eval_every(800), 100)
        self.assertEqual(planner.pick_warmup(800), 20)

    def test_warnings(self):
        many = planner.make_plan(self.S, 4096, 1, 4, passes=12)
        self.assertTrue(any("stock reply" in w for w in many["warnings"]))
        novalid = planner.make_plan(dict(self.S, val_tokens=100), 4096, 1, 4)
        self.assertTrue(any("validation" in w for w in novalid["warnings"]))
        little = planner.make_plan(dict(self.S, loss_tokens=10_000), 4096, 1, 4)
        self.assertTrue(any("rough hint" in w for w in little["warnings"]))
        tiny = planner.make_plan(dict(self.S, train_tokens=5000), 4096, 1, 4)
        self.assertTrue(any("step(s)" in w for w in tiny["warnings"]))
        pre = planner.make_plan(dict(self.S, kind="pretrain"), 4096, 1, 4)
        self.assertEqual(pre["passes"], 1.0)
        self.assertEqual(pre["lr"], 6e-4)
        self.assertEqual(planner.make_plan(self.S, 4096, 1, 4, tokens_per_sec=16384)["minutes"], 754 / 60.0)

    def test_command_turns_early_stop_and_best_on_and_parses(self):
        p = planner.make_plan(self.S, 4096, 1, 4)
        c = planner.command(p, "work3\\run_chat4", "data_chat4096", "s30m", "work3\\run_pre\\ckpt.npz")
        self.assertIn("--early-stop 3 --save-best 1", c)
        self.assertIn("--steps 754", c)
        a = cli.build_parser().parse_args(c.split()[3:])           # the printed command must be accepted by the real parser
        self.assertEqual((a.steps, a.early_stop, a.save_best, a.batch, a.accum), (754, 3, 1, 1, 4))
        self.assertEqual(a.init_from, "work3\\run_pre\\ckpt.npz")

    def test_where_it_peaked(self):
        hist = [{"step": 100, "val_loss": 1.2}, {"step": 800, "val_loss": 0.937}, {"step": 3000, "val_loss": 1.351}, {"step": 3000, "loss": 0.27}]
        r = planner.where_it_peaked(hist, 16384, 4_116_902)
        self.assertEqual((r["best_step"], r["last_step"]), (800, 3000))
        self.assertAlmostEqual(r["best_passes"], 3.18, places=1)
        self.assertIsNone(planner.where_it_peaked([{"step": 1, "loss": 1.0}], 1, 1))

    def test_command_on_a_real_token_folder_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok, cfg, data = selfcheck.make_tiny_data(tmp)
            run = os.path.join(tmp, "run")
            os.makedirs(run)
            meta = {"cfg": dict(PRESETS["tiny"]), "hp": {"batch": 1, "accum": 4, "steps": 800}, "step": 3000,
                    "history": [{"step": 100, "val_loss": 1.2}, {"step": 800, "val_loss": 0.937}, {"step": 3000, "val_loss": 1.351}]}
            meta["cfg"]["ctx"] = 16
            np.savez(os.path.join(run, "ckpt.npz"), meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
            a = cli.build_parser().parse_args(["plan-steps", "--data-dir", data, "--size", "tiny", "--ctx", "16", "--from-run", run, "--tokens-per-sec", "1000"])
            lines = []
            planner.cmd(a, say=lines.append)
            text = "\n".join(lines)
            self.assertIn("Step plan for", text)
            self.assertIn("--early-stop 3", text)
            self.assertIn("validation loss was best (0.937) at step 800", text)
            self.assertIn("overfit after that peak", text)
            self.assertIn("chat-probe", text)

    def test_not_a_token_folder_is_a_friendly_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = cli.build_parser().parse_args(["plan-steps", "--data-dir", tmp])
            with self.assertRaises(SystemExit) as cm:
                planner.cmd(a, say=lambda *_: None)
            self.assertIn("token folder", str(cm.exception))


class WindowsSelfCheck(unittest.TestCase):
    def test_a_console_that_cannot_show_boxes_gets_the_ascii_look(self):
        d = pretty.TrainDisplay("t", selfcheck.StrictStream("cp1252"), color=False)
        self.assertFalse(d.unicode)

    def test_the_check_can_catch_a_unicode_crash(self):
        with self.assertRaises(UnicodeEncodeError):
            pretty.TrainDisplay("t", selfcheck.StrictStream("cp1252"), color=False, unicode_ok=True)

    def test_a_crashing_check_becomes_a_problem_line(self):
        rep = doctor.Report(lambda *_: None)

        def boom():
            raise ValueError("nope")
        self.assertIsNone(selfcheck._guard(rep, "x", boom))
        self.assertEqual(rep.count(doctor.PROBLEM), 1)
        self.assertIn("ValueError", rep.items[0][2])
        selfcheck._guard(rep, "y", lambda: (_ for _ in ()).throw(SystemExit("bye")))       # a SystemExit inside a check does not end the self-check
        self.assertEqual(rep.count(doctor.PROBLEM), 2)

    def test_whole_check_runs_here_and_touches_nothing(self):
        with tempfile.TemporaryDirectory() as proj:
            os.makedirs(os.path.join(proj, "work"))
            with open(os.path.join(proj, "work", "keep.txt"), "w") as f:
                f.write("mine")
            lines = []
            code = selfcheck.run(proj, run_tests=False, say=lines.append)
            text = "\n".join(lines)
            self.assertEqual(code, 0, text)
            for word in ("numpy training", "chat sampling", "training panel", "menu form", "Result:"):
                self.assertIn(word, text)
            self.assertEqual(sorted(os.listdir(proj)), ["work"])
            self.assertEqual(os.listdir(os.path.join(proj, "work")), ["keep.txt"])

    def test_onedrive_and_long_paths_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            deep = os.path.join(tmp, "OneDrive", "x" * 150)
            os.makedirs(deep)
            rep = doctor.Report(lambda *_: None)
            selfcheck.check_computer(rep, deep)
            text = " ".join(i[2] for i in rep.items if i[0] == doctor.WARN)
            self.assertIn("OneDrive", text)
            self.assertIn("long", text)

    def test_doctor_flag_runs_the_selfcheck_and_the_normal_doctor_is_unchanged(self):
        a = cli.build_parser().parse_args(["doctor", "--windows", "--no-tests"])
        self.assertTrue(a.windows and a.no_tests)
        b = cli.build_parser().parse_args(["doctor"])
        self.assertFalse(b.windows)
        out = io.StringIO()
        with redirect_stdout(out):
            doctor.cmd(a)
        self.assertIn("Windows self-check", out.getvalue())
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                try:
                    doctor.cmd(SimpleNamespace(root=tmp, tokenizer=None, deep=False, windows=False, no_tests=False))
                except SystemExit:
                    pass                    # no tokenizer in an empty folder is a PROBLEM (exit code 1): that is the normal doctor
            self.assertIn("Persona_LM doctor (read-only)", out.getvalue())


class Docs(unittest.TestCase):
    def read(self, rel):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            return f.read()

    def test_quickstart_exists_and_is_linked(self):
        q = self.read(os.path.join("docs", "QUICKSTART.md"))
        self.assertIn("docs/QUICKSTART.md", self.read("README.md"))
        for needle in ("Troubleshooting", "How much data", "chat-probe", "plan-steps", "doctor --windows", "100M"):
            self.assertIn(needle, q)

    def test_every_command_in_the_quickstart_is_real(self):
        q = self.read(os.path.join("docs", "QUICKSTART.md"))
        names = set(cli.build_parser()._subparsers._group_actions[0].choices)
        used = set(re.findall(r"python -m persona_lm ([a-z][a-z0-9-]*)", q))
        self.assertGreaterEqual(len(used), 8)
        self.assertEqual(used - names, set(), "the quickstart names commands that do not exist")

    def test_every_flag_in_a_quickstart_command_is_real(self):
        q = self.read(os.path.join("docs", "QUICKSTART.md"))
        parser = cli.build_parser()
        lines = re.findall(r"^python -m persona_lm .*$", q, flags=re.M) + re.findall(r"`(python -m persona_lm [^`]*)`", q)
        self.assertGreaterEqual(len(lines), 10)
        for line in lines:
            if "<" in line or "..." in line:
                continue
            argv = line.replace("python -m persona_lm", "").split()
            try:
                with redirect_stdout(io.StringIO()):
                    parser.parse_args(argv)
            except SystemExit as e:
                self.fail("the parser rejects this quickstart command: %s (%s)" % (line, e))


if __name__ == "__main__":
    unittest.main()
