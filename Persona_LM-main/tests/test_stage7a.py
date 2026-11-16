"""Stage 7a tests: proposals (memory and Person-card edits), typed keyboard-only approval, undo, lineage entries without texts, command line.
Nothing here reads the real keyboard (a module guard makes any accidental keyboard read fail instead of wait)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import cli
from persona_lm import identity as I
from persona_lm import memory as M
from persona_lm import person as P
from persona_lm import proposals as PR

T0 = 1_700_000_000.0


def setUpModule():
    global _guard
    def no_keyboard(prompt=""):
        raise AssertionError("a test tried to read from the keyboard: %r" % prompt)
    _guard = mock.patch("builtins.input", no_keyboard)
    _guard.start()
    I.REQUIRE_TTY = False


def tearDownModule():
    _guard.stop()
    I.REQUIRE_TTY = True


def typed(*answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        self.t += 1
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.clock = Clock()
        self.mem, self.card = self.p("mem.json"), self.p("card.md")
        store = M.MemoryStore(self.mem, clock=self.clock)
        self.m1 = store.add("Maria is my sister and loves mango", importance=0.6, nodes=["person:Maria"])["id"]
        store.save()
        with open(self.card, "w", encoding="utf-8") as f:
            f.write("# Person card\n## Name\n- Jaymar\n## Habits\n- I wake up early\n")
        self.core = I.Core.create(self.p("core"), "Luna", "Jaymar", personality="curious", values="honesty", clock=self.clock)
        self.pr = self.make()

    def p(self, *parts):
        return os.path.join(self.root, *parts)

    def make(self):
        return PR.Proposals(I.Core(self.p("core"), clock=self.clock), memory_path=self.mem, person_path=self.card)

    def memory(self):
        return M.MemoryStore(self.mem, clock=self.clock)

    def out(self):
        return lambda *a: None

    def approve(self, pid):
        return self.pr.approve(pid, ask=typed(pid), out=self.out())

    def log_text(self):
        return read(self.p("core", "lineage.jsonl"))


class Submitting(Base):
    def test_filing_changes_nothing_but_the_proposal_file(self):
        before = (read(self.mem), read(self.card), self.log_text())
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "owner said so in chat")
        self.pr.submit("card-add", {"section": "habits", "text": "I read at night"}, "mentioned twice")
        self.assertEqual(p["status"], "pending")
        self.assertEqual(before, (read(self.mem), read(self.card), self.log_text()))
        self.assertTrue(os.path.exists(self.p("core", "proposals.json")))
        self.assertEqual(len(self.make().listing("pending")), 2)             # survives a reload

    def test_identity_cannot_be_proposed(self):
        for kind in ("identity-edit", "identity", "config", "finetune"):
            with self.assertRaises(PR.ProposalProblem):
                self.pr.submit(kind, {"section": "Values", "text": "x"}, "why")

    def test_needs_a_reason(self):
        with self.assertRaises(PR.ProposalProblem):
            self.pr.submit("memory-add", {"text": "x"}, "  ")
        with self.assertRaises(PR.ProposalProblem):
            self.pr.submit("memory-add", {"text": "x"}, "y" * 301)

    def test_bad_arguments_refused_at_filing(self):
        for kind, args in (("memory-add", {"text": ""}), ("memory-add", {"text": "x", "importance": 2}), ("memory-forget", {"id": "m9999"}),
                           ("memory-importance", {"id": self.m1, "importance": 1.5}), ("card-add", {"section": "Nonsense", "text": "x"}),
                           ("card-add", {"section": "Habits", "text": "x" * 500})):
            with self.assertRaises((PR.ProposalProblem, M.MemoryProblem, P.PersonProblem), msg=str((kind, args))):
                self.pr.submit(kind, args, "why")
        self.assertEqual(self.pr.listing(), [])

    def test_limit_on_waiting_proposals(self):
        for i in range(PR.MAX_PENDING):
            self.pr.submit("memory-add", {"text": "note %d" % i}, "why")
        with self.assertRaises(PR.ProposalProblem):
            self.pr.submit("memory-add", {"text": "one more"}, "why")

    def test_numbers_are_not_reused_and_get_accepts_digits(self):
        a = self.pr.submit("memory-add", {"text": "a"}, "w")
        self.pr.reject(a["id"])
        b = self.pr.submit("memory-add", {"text": "b"}, "w")
        self.assertEqual((a["id"], b["id"]), ("p0001", "p0002"))
        self.assertEqual(self.pr.get("2")["id"], "p0002")


class Approving(Base):
    def test_approve_memory_add(self):
        p = self.pr.submit("memory-add", {"text": "I like rain", "emotion": "joy", "importance": 0.7, "nodes": ["topic:weather"]}, "why")
        msg = self.approve(p["id"])
        self.assertIn("Approved p0001", msg)
        mems = [m for m in self.memory().memories.values() if m["text"] == "I like rain"]
        self.assertEqual(len(mems), 1)
        self.assertEqual((mems[0]["emotion"], mems[0]["importance"]), ("joy", 0.7))
        self.assertTrue(mems[0]["source"].startswith("ai-proposal"))
        self.assertEqual(self.make().get(p["id"])["status"], "approved")

    def test_wrong_confirmation_changes_nothing(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        before = read(self.mem), self.log_text()
        for answer in ("", "yes", "p0002", "APPROVE"):
            with self.assertRaises(PR.ProposalProblem):
                self.pr.approve(p["id"], ask=typed(answer), out=self.out())
        self.assertEqual(before, (read(self.mem), self.log_text()))
        self.assertEqual(self.pr.get(p["id"])["status"], "pending")

    def test_typed_id_is_case_and_space_tolerant(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        self.pr.approve(p["id"], ask=typed("  P0001 "), out=self.out())
        self.assertEqual(self.pr.get(p["id"])["status"], "approved")

    def test_no_double_approval(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        self.approve(p["id"])
        with self.assertRaises(PR.ProposalProblem):
            self.approve(p["id"])
        self.assertEqual(sum(1 for m in self.memory().memories.values() if m["text"] == "I like rain"), 1)

    def test_hand_edited_proposal_file_is_refused(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        path = self.p("core", "proposals.json")
        d = json.loads(read(path))
        d["items"][p["id"]]["args"]["text"] = "something else entirely"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)
        with self.assertRaises(PR.ProposalProblem) as cm:
            self.make().approve(p["id"], ask=typed(p["id"]), out=self.out())
        self.assertIn("edited by hand", str(cm.exception))
        self.assertFalse(any(m["text"] == "something else entirely" for m in self.memory().memories.values()))

    def test_approve_card_add_keeps_owner_lines(self):
        p = self.pr.submit("card-add", {"section": "habits", "text": "I read at night"}, "why")
        self.approve(p["id"])
        card = P.load(self.card)
        self.assertEqual([f["text"] for f in card.facts("Habits")], ["I wake up early", "I read at night"])
        self.assertEqual([f["text"] for f in card.facts("Name")], ["Jaymar"])

    def test_approve_importance_and_pin(self):
        p = self.pr.submit("memory-importance", {"id": self.m1, "importance": 0.95, "pinned": True}, "why")
        self.approve(p["id"])
        m = self.memory().get(self.m1)
        self.assertEqual((m["importance"], m["pinned"]), (0.95, True))

    def test_approve_forget(self):
        p = self.pr.submit("memory-forget", {"id": self.m1}, "duplicate")
        self.approve(p["id"])
        self.assertNotIn(self.m1, self.memory().memories)
        self.assertNotIn("mango", read(self.mem))
        prev = self.mem[:-5] + ".prev.json"
        self.assertFalse(os.path.exists(prev) and "mango" in read(prev))     # forgotten means gone from the backup too

    def test_target_gone_at_approval_time(self):
        p = self.pr.submit("memory-forget", {"id": self.m1}, "why")
        s = self.memory()
        s.forget(self.m1)
        s.save()
        with self.assertRaises((PR.ProposalProblem, M.MemoryProblem)):
            self.approve(p["id"])
        self.assertEqual(self.pr.get(p["id"])["status"], "pending")

    def test_conflict_with_another_program_leaves_proposal_pending(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        orig = M.MemoryStore.save
        def clash(self_, *a, **k):
            raise M.MemoryConflict("changed by another program")
        with mock.patch.object(M.MemoryStore, "save", clash), self.assertRaises(M.MemoryConflict):
            self.approve(p["id"])
        self.assertEqual(self.make().get(p["id"])["status"], "pending")
        self.assertEqual(len(self.make().core.lineage.entries()), 1)         # only the genesis line

    def test_reject(self):
        p = self.pr.submit("card-add", {"section": "Habits", "text": "x"}, "why")
        before = read(self.card)
        self.assertIn("Rejected", self.pr.reject(p["id"], "not true"))
        self.assertEqual((before, self.pr.get(p["id"])["status"], self.pr.get(p["id"])["note"]), (read(self.card), "rejected", "not true"))
        with self.assertRaises(PR.ProposalProblem):
            self.approve(p["id"])


class Keyboard(Base):
    def test_real_keyboard_rule_refuses_pipes(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        before = read(self.mem)
        I.REQUIRE_TTY = True
        try:
            with mock.patch.object(sys, "stdin", io.StringIO("p0001\n")), self.assertRaises(I.IdentityProblem):
                self.pr.approve(p["id"], out=self.out())                    # default ask = keyboard_ask
        finally:
            I.REQUIRE_TTY = False
        self.assertEqual((before, self.pr.get(p["id"])["status"]), (read(self.mem), "pending"))
        self.approve(p["id"])                                               # typed properly (test keyboard) it works; now undo from a pipe
        after = read(self.mem)
        I.REQUIRE_TTY = True
        try:
            with mock.patch.object(sys, "stdin", io.StringIO("p0001\n")), self.assertRaises(I.IdentityProblem):
                self.pr.undo(p["id"], out=self.out())
            with mock.patch.object(sys, "stdin", io.StringIO("p0001\n")), self.assertRaises(I.IdentityProblem):
                self.pr.purge_undo(p["id"])
        finally:
            I.REQUIRE_TTY = False
        self.assertEqual((after, self.pr.get(p["id"])["status"]), (read(self.mem), "approved"))

    def test_proposals_module_never_reads_input_by_itself(self):
        src = read(os.path.join(os.path.dirname(PR.__file__), "proposals.py"))
        self.assertNotIn("input(", src.replace("keyboard_ask", ""))


class Undoing(Base):
    def test_undo_memory_add(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        self.approve(p["id"])
        self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())
        self.assertFalse(any(m["text"] == "I like rain" for m in self.memory().memories.values()))
        self.assertEqual(self.pr.get(p["id"])["status"], "undone")

    def test_undo_memory_add_refused_when_changed_since(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        self.approve(p["id"])
        s = self.memory()
        mid = [m["id"] for m in s.memories.values() if m["text"] == "I like rain"][0]
        s.memories[mid]["text"] = "I like rain a lot"
        s.save()
        with self.assertRaises(PR.ProposalProblem):
            self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())
        self.assertEqual(self.pr.get(p["id"])["status"], "approved")

    def test_undo_forget_restores_memory_and_nodes(self):
        before = self.memory().get(self.m1)
        p = self.pr.submit("memory-forget", {"id": self.m1}, "why")
        self.approve(p["id"])
        self.assertEqual(self.memory().nodes, {})
        self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())
        s = self.memory()
        self.assertEqual(s.get(self.m1), before)
        self.assertTrue(s.find_node("Maria"))

    def test_purge_undo_erases_the_copy(self):
        p = self.pr.submit("memory-forget", {"id": self.m1}, "why")
        self.approve(p["id"])
        self.assertIn("mango", read(self.p("core", "proposals.json")))       # honest: the copy lives here until purged
        self.pr.purge_undo(p["id"], ask=typed(p["id"]))
        self.assertNotIn("mango", read(self.p("core", "proposals.json")).replace(self.pr.get(p["id"])["why"], ""))
        with self.assertRaises(PR.ProposalProblem):
            self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())

    def test_undo_importance(self):
        p = self.pr.submit("memory-importance", {"id": self.m1, "importance": 0.95, "pinned": True}, "why")
        self.approve(p["id"])
        self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())
        m = self.memory().get(self.m1)
        self.assertEqual((m["importance"], bool(m["pinned"])), (0.6, False))

    def test_undo_card_add_only_removes_that_line(self):
        p = self.pr.submit("card-add", {"section": "Habits", "text": "I read at night"}, "why")
        self.approve(p["id"])
        self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())
        self.assertEqual([f["text"] for f in P.load(self.card).facts("Habits")], ["I wake up early"])

    def test_undo_card_add_refused_when_line_was_edited_by_hand(self):
        p = self.pr.submit("card-add", {"section": "Habits", "text": "I read at night"}, "why")
        self.approve(p["id"])
        with open(self.card, "w", encoding="utf-8") as f:
            f.write(read(self.card).replace("I read at night", "I read in the morning"))
        with self.assertRaises(PR.ProposalProblem):
            self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())

    def test_only_approved_can_be_undone_and_wrong_typing_cancels(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        with self.assertRaises(PR.ProposalProblem):
            self.pr.undo(p["id"], ask=typed(p["id"]), out=self.out())
        self.approve(p["id"])
        with self.assertRaises(PR.ProposalProblem):
            self.pr.undo(p["id"], ask=typed("no"), out=self.out())
        self.assertEqual(self.pr.get(p["id"])["status"], "approved")


class LineageAndIdentity(Base):
    def test_every_decision_is_logged_without_texts(self):
        a = self.pr.submit("memory-add", {"text": "My secret plan is mango"}, "why")
        b = self.pr.submit("card-add", {"section": "Habits", "text": "private habit line"}, "why")
        c = self.pr.submit("memory-forget", {"id": self.m1}, "why")
        self.approve(a["id"])
        self.pr.reject(b["id"])
        self.approve(c["id"])
        self.pr.undo(a["id"], ask=typed(a["id"]), out=self.out())
        kinds = [e["kind"] for e in self.core.lineage.entries()]
        self.assertEqual(kinds, ["genesis", "proposal-approved", "proposal-rejected", "proposal-approved", "proposal-undone"])
        log = read(self.p("core", "lineage.jsonl"))
        for secret in ("secret plan", "private habit", "mango"):
            self.assertNotIn(secret, log)
        self.assertEqual(I.Core(self.p("core")).lineage.verify()[0], True)
        self.assertTrue(all(e["approved_by"] == PR.APPROVER for e in self.core.lineage.entries()[1:]))

    def test_core_head_follows_so_cutting_the_log_is_noticed(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        self.approve(p["id"])
        core = I.Core(self.p("core"))
        self.assertEqual(core.d["lineage_head"], core.lineage.head())
        self.assertEqual(core.d["lineage_head"]["n"], 2)

    def test_damaged_log_blocks_approval(self):
        p = self.pr.submit("memory-add", {"text": "I like rain"}, "why")
        lg = self.p("core", "lineage.jsonl")
        text = read(lg).replace("identity core created", "identity core CHANGED")
        self.assertIn("CHANGED", text)
        with open(lg, "w", encoding="utf-8") as f:
            f.write(text)
        before = read(self.mem)
        with self.assertRaises(PR.ProposalProblem):
            self.make().approve(p["id"], ask=typed(p["id"]), out=self.out())
        self.assertEqual(before, read(self.mem))

    def test_identity_core_text_untouched_by_proposals(self):
        before = dict(self.core.sections)
        for args in ({"text": "my name is Bob"}, {"text": "values: cruelty"}):
            p = self.pr.submit("memory-add", args, "why")
            self.approve(p["id"])
        self.assertEqual(before, I.Core(self.p("core")).sections)


class CommandLine(Base):
    def run_cli(self, *argv, keys=None):
        buf = io.StringIO()
        ctx = mock.patch("builtins.input", typed(*keys)) if keys is not None else contextlib.nullcontext()
        with ctx, contextlib.redirect_stdout(buf):
            cli.main(["proposal", "--dir", self.p("core"), "--memory", self.mem, "--person", self.card] + list(argv))
        return buf.getvalue()

    def fails(self, *argv, keys=None):
        buf = io.StringIO()
        ctx = mock.patch("builtins.input", typed(*keys)) if keys is not None else contextlib.nullcontext()
        with ctx, contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            cli.main(["proposal", "--dir", self.p("core"), "--memory", self.mem, "--person", self.card] + list(argv))
        return str(cm.exception) + buf.getvalue()

    def test_file_list_show_approve_undo(self):
        out = self.run_cli("memory-add", "I", "like", "rain", "--why", "said it today", "--emotion", "joy")
        self.assertIn("Nothing was changed", out)
        self.assertIn("p0001", self.run_cli("list"))
        self.assertIn("+ new memory: I like rain", self.run_cli("show", "p0001"))
        self.assertIn("Approved p0001", self.run_cli("approve", "p0001", keys=["p0001"]))
        self.assertTrue(any(m["text"] == "I like rain" for m in self.memory().memories.values()))
        self.assertIn("Undone p0001", self.run_cli("undo", "p0001", keys=["p0001"]))
        self.assertIn("undone", self.run_cli("list", "--all"))

    def test_card_importance_forget_commands(self):
        self.run_cli("card-add", "Habits", "I", "read", "--why", "w")
        self.run_cli("memory-importance", self.m1, "0.9", "--why", "w", "--pinned")
        self.run_cli("memory-forget", self.m1, "--why", "w")
        self.assertEqual(len(self.make().listing("pending")), 3)
        self.run_cli("reject", "p0003", "--note", "keep it")
        self.assertIn("rejected", self.run_cli("list", "--all"))

    def test_approve_from_a_pipe_is_refused(self):
        self.run_cli("memory-add", "I", "like", "rain", "--why", "w")
        before = read(self.mem)
        I.REQUIRE_TTY = True
        try:
            with mock.patch.object(sys, "stdin", io.StringIO("p0001\n")):
                msg = self.fails("approve", "p0001")
        finally:
            I.REQUIRE_TTY = False
        self.assertIn("keyboard", msg)
        self.assertEqual(before, read(self.mem))

    def test_errors_are_plain_messages(self):
        self.assertIn("no proposal", self.fails("show", "p0099"))
        self.assertIn("Say why", self.fails("memory-add", "x", "--why", " "))
        self.assertIn("There is no memory", self.fails("memory-forget", "m9999", "--why", "w"))

    def test_uses_files_registered_in_the_core(self):
        c = I.Core(self.p("core"))
        c.paths_set("memory", self.mem)
        c.paths_set("person", self.card)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["proposal", "--dir", self.p("core"), "memory-add", "hello", "world", "--why", "w"])
        self.assertIn("Filed as p0001", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
