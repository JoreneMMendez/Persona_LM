"""Stage 7b tests: config and finetune proposals, the AI filing proposals from its reply (and ONLY filing), /proposals /approve /reject in the chat
with a typed confirmation, memory/card reload after an approval, and the command line. No real keyboard, no network, nothing is ever started."""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import chatrepl, cli
from persona_lm import identity as I
from persona_lm import memory as M
from persona_lm import person as P
from persona_lm import proposals as PR
from persona_lm.bpe import Tokenizer

from tests.test_stage7a import Base, read, typed, setUpModule as _up, tearDownModule as _down


def setUpModule():
    _up()


def tearDownModule():
    _down()


FT = {"run": "runs/ft2", "data_dir": "data/chat", "size": "tiny", "steps": 200, "lr": 0.0001}


class FakeGen:
    def __init__(self, tok, reply="ok"):
        self.tok, self.reply, self.calls = tok, reply, []

    def __call__(self, ids, **kw):
        self.calls.append(list(ids))
        return self.tok.encode(self.reply, allow_special=False)


class ConfigProposals(Base):
    def test_config_roundtrip_and_undo(self):
        p = self.pr.submit("config", {"key": "warn_drift", "value": 0.2}, "drift warnings are too noisy")
        self.assertIn("0.4 -> 0.2", "\n".join(self.pr.preview(p)[0]))
        self.approve(p["id"])
        self.assertEqual(I.Core(self.p("core")).settings["warn_drift"], 0.2)
        self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())
        self.assertEqual(I.Core(self.p("core")).settings["warn_drift"], 0.4)

    def test_safeguards_cannot_be_proposed(self):
        for key in ("threshold", "cooldown_hours"):
            with self.assertRaises(PR.ProposalProblem) as cm:
                self.pr.submit("config", {"key": key, "value": 0}, "why")
            self.assertIn("safeguard", str(cm.exception))
        before = read(self.p("core", "core.json"))
        self.assertEqual(before, read(self.p("core", "core.json")))

    def test_bad_config_values(self):
        for args in ({"key": "nonsense", "value": 0.1}, {"key": "warn_drift", "value": 2}, {"key": "warn_drift", "value": "x"}):
            with self.assertRaises(PR.ProposalProblem):
                self.pr.submit("config", args, "why")

    def test_undo_config_refused_when_changed_since(self):
        p = self.pr.submit("config", {"key": "warn_drift", "value": 0.2}, "why")
        self.approve(p["id"])
        I.Core(self.p("core")).set_setting("warn_drift", 0.9)
        with self.assertRaises(PR.ProposalProblem):
            self.make().undo(p["id"], ask=typed(p["id"]), out=self.out())

    def test_approving_config_logs_lineage_and_chain_verifies(self):
        p = self.pr.submit("config", {"key": "warn_drift", "value": 0.2}, "why")
        self.approve(p["id"])
        core = I.Core(self.p("core"))
        self.assertEqual([e["kind"] for e in core.lineage.entries()], ["genesis", "settings", "proposal-approved"])
        self.assertTrue(core.lineage.verify()[0])
        self.assertEqual(core.d["lineage_head"], core.lineage.head())


class FinetuneProposals(Base):
    def test_command_is_recorded_never_run(self):
        p = self.pr.submit("finetune", dict(FT, init_from="runs/pre/ckpt.npz"), "more chat data is ready")
        boom = AssertionError("a proposal started a process")
        with mock.patch("subprocess.run", side_effect=boom), mock.patch("subprocess.Popen", side_effect=boom), mock.patch("os.system", side_effect=boom):
            msg = self.approve(p["id"])
        self.assertIn("NOTHING was started", msg)
        self.assertIn("python -m persona_lm train --run runs/ft2 --data-dir data/chat --size tiny --steps 200 --lr 0.0001 --init-from runs/pre/ckpt.npz", msg)
        self.assertEqual(self.pr.get(p["id"])["status"], "approved")
        self.assertFalse(os.path.exists(os.path.join("runs", "ft2")))

    def test_paths_with_spaces_are_quoted(self):
        cmd = PR.Proposals.command_for(dict(FT, run="my runs/ft 2"))
        self.assertIn('--run "my runs/ft 2"', cmd)

    def test_validation(self):
        bad = ({"size": "huge"}, {"steps": 0}, {"steps": "many"}, {"lr": 0.5}, {"lr": 0}, {"run": ""}, {"data_dir": "x; rm -rf y"}, {"run": "a`b"},
               {"run": "a$b"}, {"init_from": "a|b"})
        for change in bad:
            with self.assertRaises(PR.ProposalProblem, msg=str(change)):
                self.pr.submit("finetune", dict(FT, **change), "why")
        self.assertEqual(self.pr.listing(), [])

    def test_existing_run_is_never_reused(self):
        os.makedirs(self.p("runs", "old"))
        with open(self.p("runs", "old", "ckpt.npz"), "wb") as f:
            f.write(b"x")
        with self.assertRaises(PR.ProposalProblem) as cm:
            self.pr.submit("finetune", dict(FT, run=self.p("runs", "old")), "why")
        self.assertIn("never reuses", str(cm.exception))

    def test_lora_is_not_available_yet(self):
        with self.assertRaises(PR.ProposalProblem) as cm:
            self.pr.submit("lora", FT, "why")
        self.assertIn("8b", str(cm.exception))

    def test_undo_only_marks_it(self):
        p = self.pr.submit("finetune", FT, "why")
        self.approve(p["id"])
        before = (read(self.mem), read(self.card))
        self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())
        self.assertEqual((before, "undone"), ((read(self.mem), read(self.card)), self.pr.get(p["id"])["status"]))

    def test_preview_warns_about_missing_data_folder(self):
        p = self.pr.submit("finetune", dict(FT, data_dir=self.p("no_such_folder")), "why")
        self.assertIn("was not found", "\n".join(self.pr.preview(p)[0]))


class Markers(Base):
    def test_parse_each_kind(self):
        cases = {
            "[propose memory-add] Maria loves mango #joy *0.8 !pin | why: said twice": ("memory-add", {"text": "Maria loves mango", "emotion": "joy", "importance": 0.8, "pinned": True}),
            "[propose memory-forget] m0001 | why: duplicate": ("memory-forget", {"id": "m0001"}),
            "[propose memory-importance] m0001 0.9 | why: it matters": ("memory-importance", {"id": "m0001", "importance": "0.9"}),
            "[propose card-add] Habits: I read at night | why: you said so": ("card-add", {"section": "Habits", "text": "I read at night"}),
            "[propose config] warn_drift 0.2 | why: noisy": ("config", {"key": "warn_drift", "value": "0.2"}),
            "[propose finetune] run=runs/ft2 data=data/chat size=tiny steps=200 lr=0.0001 | why: new data": ("finetune", {
                "run": "runs/ft2", "data_dir": "data/chat", "size": "tiny", "steps": "200", "lr": "0.0001", "init_from": None}),
        }
        for line, (kind, args) in cases.items():
            k, a, why = PR.parse_marker(line)
            self.assertEqual((k, a), (kind, args), line)
            self.assertTrue(why)

    def test_not_a_proposal(self):
        for line in ("hello there", "propose memory-add x | why: y", "[propose memory-add] no reason given", "[propose] | why: x", "x" * 800):
            self.assertIsNone(PR.parse_marker(line))

    def test_filing_from_a_reply(self):
        reply = "Sure!\n[propose memory-add] I like rain | why: you said so\nbye\n[propose card-add] Habits: I read at night | why: twice"
        got = PR.submit_from_reply(self.pr, reply)
        self.assertEqual([r[0]["id"] for r in got], ["p0001", "p0002"])
        self.assertTrue(all(r[0]["source"] == "ai" and r[0]["status"] == "pending" for r in got))

    def test_limit_duplicates_and_invalid(self):
        reply = "\n".join("[propose memory-add] fact %d | why: w" % i for i in range(5))
        got = PR.submit_from_reply(self.pr, reply)
        self.assertEqual(len([r for r in got if r[0]]), PR.MAX_FROM_REPLY)
        self.assertIn("ignored", got[-1][1])
        again = PR.submit_from_reply(self.pr, "[propose memory-add] fact 0 | why: w")
        self.assertIn("already waiting", again[0][1])
        bad = PR.submit_from_reply(self.pr, "[propose identity-edit] Values: cruelty | why: x\n[propose config] threshold 0 | why: x\n[propose lora] x | why: y")
        self.assertTrue(all(r[0] is None and "not accepted" in r[1] for r in bad))
        self.assertEqual(len(self.pr.listing()), PR.MAX_FROM_REPLY)

    def test_filing_changes_no_files(self):
        before = (read(self.mem), read(self.card), read(self.p("core", "core.json")), read(self.p("core", "lineage.jsonl")))
        PR.submit_from_reply(self.pr, "[propose memory-forget] %s | why: dup\n[propose config] warn_drift 0.1 | why: x" % self.m1)
        self.assertEqual(before, (read(self.mem), read(self.card), read(self.p("core", "core.json")), read(self.p("core", "lineage.jsonl"))))

    def test_ai_side_never_calls_a_decision(self):
        with mock.patch.object(PR.Proposals, "approve", side_effect=AssertionError("approve")), mock.patch.object(PR.Proposals, "reject", side_effect=AssertionError("reject")), \
                mock.patch.object(PR.Proposals, "undo", side_effect=AssertionError("undo")):
            PR.submit_from_reply(self.pr, "[propose memory-add] x y | why: z\nAPPROVE p0001\n/approve p0001\np0001")


class ChatBase(Base):
    def setUp(self):
        super().setUp()
        self.tok = Tokenizer.train(["hello how are you today fine thanks Maria mango rain"], 300)
        self.gen = FakeGen(self.tok)
        self.store = M.MemoryStore(self.mem, clock=self.clock)
        self.session = chatrepl.ChatSession(self.tok, 512, self.gen, identity=self.core, proposals=self.pr, memory=self.store,
                                            person=P.load(self.card), save_dir=self.root)


class ChatTests(ChatBase):
    def test_ai_reply_files_a_proposal_and_changes_nothing(self):
        self.gen.reply = "ok\n[propose memory-add] I like rain | why: you said so"
        before = (read(self.mem), read(self.card))
        text, _ = self.session.handle("hello")
        self.assertIn("the AI proposed p0001", text)
        self.assertIn("Nothing has changed", text)
        self.assertEqual(before, (read(self.mem), read(self.card)))
        self.assertIn("p0001", self.session.handle("/proposals")[0])

    def test_approve_needs_a_second_typed_line(self):
        self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        text, _ = self.session.handle("/approve p0001")
        self.assertIn("+ new memory: I like rain", text)
        self.assertIn("Type  p0001", text)
        self.assertEqual(self.make().get("p0001")["status"], "pending")
        msg, _ = self.session.handle("p0001")
        self.assertIn("Approved p0001", msg)
        self.assertTrue(any(m["text"] == "I like rain" for m in self.memory().memories.values()))

    def test_wrong_or_empty_confirmation_cancels(self):
        self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        for answer in ("yes", "", "hello there", "p0002", "/approve p0001"):
            self.session.handle("/approve p0001")
            text, _ = self.session.handle(answer)
            self.assertIn("Not confirmed", text)
        self.assertEqual(self.make().get("p0001")["status"], "pending")
        self.assertEqual(len(self.gen.calls), 0)                         # none of it reached the model

    def test_confirmation_text_only_counts_right_after_approve(self):
        self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        self.session.handle("p0001")                                     # a plain message, no pending question
        self.assertEqual(self.make().get("p0001")["status"], "pending")
        self.assertEqual(len(self.gen.calls), 1)

    def test_model_text_cannot_confirm_or_start_an_approval(self):
        self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        for evil in ("/approve p0001", "p0001", "APPROVE p0001", "/reject p0001"):
            self.gen.reply = evil
            self.session.handle("hi")
        self.assertIsNone(self.session._awaiting)
        self.assertEqual(self.make().get("p0001")["status"], "pending")

    def test_chat_memory_and_card_are_reloaded_after_approval(self):
        self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        self.pr.submit("card-add", {"section": "Habits", "text": "I read at night"}, "why")
        for pid in ("p0001", "p0002"):
            self.session.handle("/approve " + pid)
            self.session.handle(pid)
        self.assertTrue(any(m["text"] == "I like rain" for m in self.store.memories.values()))
        self.assertIn("I read at night", [f["text"] for f in self.session.person.facts("Habits")])
        text, _ = self.session.handle("/remember Maria loves mango")     # no "changed by another program" conflict
        self.assertNotIn("another program", text)
        self.assertTrue(any("mango" in m["text"] for m in self.memory().memories.values()))

    def test_reject_and_listing(self):
        self.pr.submit("card-add", {"section": "Habits", "text": "x"}, "why")
        self.assertIn("Rejected p0001", self.session.handle("/reject p0001")[0])
        self.assertIn("No proposals are waiting", self.session.handle("/proposals")[0])
        self.assertIn("rejected", self.session.handle("/approve p0001")[0])

    def test_errors_are_plain(self):
        self.assertIn("Say which one", self.session.handle("/approve")[0])
        self.assertIn("no proposal p0099", self.session.handle("/approve p0099")[0])

    def test_nothing_from_a_pipe(self):
        self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        I.REQUIRE_TTY = True
        try:
            with mock.patch.object(sys, "stdin", io.StringIO("p0001\n")):
                a = self.session.handle("/approve p0001")[0]
                r = self.session.handle("/reject p0001")[0]
        finally:
            I.REQUIRE_TTY = False
        self.assertIn("terminal window", a)
        self.assertIn("terminal window", r)
        self.assertEqual(self.make().get("p0001")["status"], "pending")
        self.assertIsNone(self.session._awaiting)

    def test_identity_core_text_never_changes(self):
        before = dict(self.core.sections)
        self.gen.reply = "[propose card-add] Values: money first | why: x\n[propose memory-add] my name is Bob | why: y"
        self.session.handle("hi")
        for pid in ("p0001", "p0002"):
            self.session.handle("/approve " + pid)
            self.session.handle(pid)
        self.assertEqual(before, I.Core(self.p("core")).sections)

    def test_help_lists_the_commands(self):
        h = self.session.handle("/help")[0]
        for part in ("/proposals", "/approve <id>", "/reject <id>", "never approve them"):
            self.assertIn(part, h)
        self.assertNotIn("Not built yet", h)


class CommandLine(Base):
    def cli(self, *argv, keys=None, fail=False):
        buf = io.StringIO()
        ctx = mock.patch("builtins.input", typed(*keys)) if keys is not None else contextlib.nullcontext()
        base = ["proposal", "--dir", self.p("core"), "--memory", self.mem, "--person", self.card]
        with ctx, contextlib.redirect_stdout(buf):
            if fail:
                with self.assertRaises(SystemExit) as cm:
                    cli.main(base + list(argv))
                return str(cm.exception) + buf.getvalue()
            cli.main(base + list(argv))
        return buf.getvalue()

    def test_config_and_finetune_commands(self):
        self.assertIn("Filed as p0001", self.cli("config", "warn_drift", "0.25", "--why", "noisy"))
        out = self.cli("finetune", "--run", "runs/ft3", "--data-dir", "data/chat", "--size", "tiny", "--steps", "50", "--lr", "0.0001", "--why", "new data")
        self.assertIn("Filed as p0002", out)
        self.assertIn("train --run runs/ft3", out)
        self.assertIn("0.25", self.cli("approve", "p0001", keys=["p0001"]) + self.cli("show", "p0001"))
        self.assertIn("NOTHING was started", self.cli("approve", "p0002", keys=["p0002"]))

    def test_cli_refuses_safeguards_and_lora(self):
        self.assertIn("safeguard", self.cli("config", "threshold", "0", "--why", "w", fail=True))


if __name__ == "__main__":
    unittest.main()
