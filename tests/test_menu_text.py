"""T1: the plain-words text bank (persona_lm/menu_text.py) covers every menu item."""
import re
import unittest

from persona_lm import menu_text as mt
import io
import os
import tempfile

from persona_lm import menu as M
from persona_lm import pretty as P
from persona_lm.menu import ITEMS, plain_text

KEYS = ("title", "short", "what", "when", "need", "result", "time", "safe")
MAX_WORDS = 18


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def all_texts():
    for label, rec in mt.ITEM_TEXT.items():
        for k in KEYS:
            yield "ITEM_TEXT[%s].%s" % (label, k), rec[k]
    for term, text in mt.GLOSSARY.items():
        yield "GLOSSARY[%s]" % term, text
    for k, text in list(mt.NEXT.items()) + list(mt.SCREEN.items()):
        yield "T2[%s]" % k, text
    for head, _ in mt.GROUPS:
        yield "GROUPS[%s]" % head, head
    for i, (_pat, what, why, todo) in enumerate(mt.FRIENDLY_ERRORS):
        for name, text in (("what", what), ("why", why), ("todo", todo)):
            yield "FRIENDLY_ERRORS[%d].%s" % (i, name), text


class TextBank(unittest.TestCase):
    def test_every_item_has_every_key(self):
        for item in ITEMS:
            rec = plain_text(item)
            for k in KEYS:
                self.assertTrue(str(rec.get(k, "")).strip(), "%s: missing %s" % (item["label"], k))

    def test_no_orphan_records(self):
        self.assertEqual(set(mt.ITEM_TEXT), {i["label"] for i in ITEMS})

    def test_sentences_are_short(self):
        for where, text in all_texts():
            for s in sentences(text):
                self.assertLessEqual(len(s.split()), MAX_WORDS, "%s: %r" % (where, s))

    def test_glossary_terms_used_are_defined(self):
        defined = " | ".join(t.lower() for t in mt.GLOSSARY)
        for where, text in all_texts():
            low = text.lower()
            for word in mt.JARGON:
                if word in low:
                    self.assertIn(word, defined, "%s uses %r but the glossary does not define it" % (where, word))

    def test_glossary_has_the_planned_terms(self):
        low = " | ".join(t.lower() for t in mt.GLOSSARY)
        for need in ("model file", "tokenizer", "prepared text files", "run", "base model", "fine-tune",
                     "test score", "overfitting", "batch", "accum", "learning speed", "gpu", "cpu", "torch", "recipe"):
            self.assertIn(need, low)

    def test_friendly_errors_cover_the_planned_cases(self):
        samples = ["CUDA out of memory", "FileNotFoundError: x", "KeyError: 'merges'", "No module named 'torch'",
                   "OSError: [Errno 28] No space left on device", "URLError: getaddrinfo failed", "KeyboardInterrupt"]
        for text in samples:
            hits = [e for e in mt.FRIENDLY_ERRORS if re.search(e[0], text, re.I)]
            self.assertTrue(hits, "no friendly error for " + text)
        for pat, what, why, todo in mt.FRIENDLY_ERRORS:
            re.compile(pat)
            self.assertTrue(what and why and todo)

    def test_text_is_plain_ascii_safe(self):
        for where, text in all_texts():
            text.encode("cp1252")


class TTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


class StrictCp1252(io.StringIO):
    encoding = "cp1252"

    def write(self, text):
        text.encode("cp1252")
        return super().write(text)

    def isatty(self):
        return True


def run(path, step, steps):
    return dict(path=path, ckpt=path + "/ckpt.npz", size="s30m", step=step, steps=steps)


STATES = {
    "empty": dict(runs=[], data=[], toks=[], torch=False),
    "data": dict(runs=[], data=["data_pre"], toks=["tok.json"], torch=True),
    "half": dict(runs=[run("work3/run_pre", 50, 100)], data=["data_pre"], toks=[], torch=True),
    "base": dict(runs=[run("work3/run_pre", 100, 100)], data=["data_pre"], toks=[], torch=True),
    "chat": dict(runs=[run("work3/run_chat2", 800, 800), run("work3/run_pre", 100, 100)], data=["a", "b"], toks=[], torch=True),
    "other": dict(runs=[run("work3/myrun", 10, 10)], data=[], toks=[], torch=False),
}


class MainScreen(unittest.TestCase):
    def test_every_item_is_in_exactly_one_group_and_quit_is_last(self):
        shown = [i for _, idxs in M.LAYOUT for i in idxs]
        self.assertEqual(sorted(shown), list(range(len(ITEMS))))
        self.assertEqual(M.DISPLAY[-1], M.QUIT)
        self.assertEqual([h for h, _ in M.LAYOUT[:-1]], ["START HERE", "TALK TO YOUR AI", "TEACH YOUR AI", "PREPARE DATA", "CHECKS AND TOOLS"])
        self.assertEqual(set(mt.TECH_NAMES), {i["label"] for i in ITEMS if i["cmd"]})

    def test_items_keep_their_old_order(self):
        self.assertEqual((ITEMS[0]["cmd"], ITEMS[1]["cmd"], ITEMS[M.QUIT]["cmd"]), ("train", "chat", None))

    def test_number_keys_pick_the_item_shown(self):
        for i in range(len(ITEMS) - 1):
            n = int(M.number_key(i))
            if n <= 9:
                self.assertEqual(M.Menu(TTY(), color=False, width=100).select(iter([str(n)]).__next__, STATES["empty"]), i)

    def test_both_screens_fit_60_and_80_columns(self):
        for width in (60, 80):
            for state in STATES.values():
                for kw in ({"color": False}, {"color": True}, {"color": False, "unicode_ok": False}):
                    lines = M.Menu(TTY(), width=width, **kw).render(0, state)
                    self.assertEqual(len(lines), M.MENU_H)
                    self.assertEqual(len({P.vlen(x) for x in lines}), 1)
                    self.assertLessEqual(P.vlen(lines[0]), width - 1)
                out = []
                M._fallback(state, lambda q: "q", out.append, width)
                for line in "\n".join(out).split("\n"):
                    self.assertLessEqual(len(line), width - 1, line)

    def test_details_show_all_six_fields_without_cutting_at_80(self):
        for i, it in enumerate(ITEMS):
            text = "\n".join(M.Menu(TTY(), color=False, width=80).render(i, STATES["empty"]))
            for lab, _ in mt.DETAIL_FIELDS:
                self.assertIn(lab, text, it["label"])
            self.assertIn(plain_text(it)["safe"].split()[-1].strip("."), text, it["label"])     # the end of the last field is not cut

    def test_next_suggestion_for_each_state(self):
        want = {"empty": "Download public text", "data": "first run", "half": "50%", "base": "cannot chat", "chat": "run_chat2", "other": "myrun"}
        for name, state in STATES.items():
            sug = M.next_suggestion(state)
            self.assertIn(want[name], sug, name)
            lines = M.Menu(TTY(), color=False, width=80).render(0, state)
            joined = " ".join(x.strip("|│ ") for x in lines[M.TOP_H:])
            self.assertIn(sug.split(".")[-2].split()[-1], joined, name)               # the sentence is not cut off

    def test_project_checks_in_words(self):
        empty = {a: b for a, b, _ in M.project_checks(STATES["empty"])}
        self.assertIn("none yet", empty["Your AI model:"])
        self.assertIn("not found", empty["Graphics card tools (PyTorch):"])
        ready = {a: b for a, b, _ in M.project_checks(STATES["chat"])}
        self.assertIn("run_chat2", ready["Your AI model:"])
        self.assertIn("+1 more", ready["Your AI model:"])

    def test_cp1252_console_and_numbered_list(self):
        out = StrictCp1252()
        M.Menu(out, color=False, width=80, unicode_ok=False).render(0, STATES["chat"])
        said = []
        M._fallback(STATES["chat"], lambda q: "2", said.append, 60)
        text = "\n".join(said)
        text.encode("cp1252")
        for head, _ in mt.GROUPS:
            self.assertIn(head, text)
        self.assertEqual(M._fallback(STATES["chat"], lambda q: "3", lambda *_: None, 60), M.DISPLAY[2])

if __name__ == "__main__":
    unittest.main()
