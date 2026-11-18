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
    for label, fields in mt.FIELD_TEXT.items():
        for key, rec in fields.items():
            for part in ("label", "help", "more", "example", "default"):
                if part not in ("example", "default"):
                    yield "FIELD_TEXT[%s][%s].%s" % (label, key, part), rec[part]
    for name, text in list(mt.FORM.items()) + list(mt.PROBLEMS.items()):
        yield "T3[%s]" % name, text
    for key, rec in mt.SETTING_TEXT.items():
        for part, text in rec.items():
            yield "SETTING_TEXT[%s].%s" % (key, part), text
    for name, why in mt.PRESET_TEXT:
        yield "PRESET_TEXT[%s].name" % name, name
        yield "PRESET_TEXT[%s].why" % name, why
    for name, text in list(mt.SETTINGS_UI.items()) + list(mt.CONFIRM.items()):
        if isinstance(text, str):
            yield "T4[%s]" % name, text
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


class Forms(unittest.TestCase):
    FOUND = dict(runs=[run("work3/run_chat3", 10, 10)], data=["work3/data_chat"], toks=["work/tok.json"], torch=True)

    def test_every_field_has_label_help_example_and_more(self):
        for item in ITEMS:
            for f in item["fields"]:
                rec = mt.FIELD_TEXT[item["label"]][f["key"]]
                for part in ("label", "help", "more", "example"):
                    self.assertTrue(rec[part].strip(), "%s/%s: no %s" % (item["label"], f["key"], part))
                self.assertLessEqual(len(rec["label"]), 28, rec["label"])
        known = {(i["label"], f["key"]) for i in ITEMS for f in i["fields"]}
        self.assertEqual({(l, k) for l, fs in mt.FIELD_TEXT.items() for k in fs}, known)

    def test_forms_fit_60_and_80_columns_even_on_a_cp1252_console(self):
        for width in (60, 80):
            for item in ITEMS:
                if not item["fields"]:
                    continue
                form = M.Form(item, self.FOUND, StrictCp1252(), color=False, width=width, unicode_ok=False)
                for step in range(len(item["fields"]) + 1):
                    for long in (False, True):
                        form.long = long
                        for line in form.lines():
                            line.encode("cp1252")
                            self.assertLessEqual(len(line), width - 1, (item["label"], line))
                    form.key("down")

    def test_question_shows_help_example_and_what_enter_does(self):
        item = ITEMS[2]                                                 # fine-tune
        form = M.Form(item, self.FOUND, TTY(), color=False, width=80)
        form.focus = 1
        text = "\n".join(form.lines())
        rec = mt.FIELD_TEXT[item["label"]]["run"]
        self.assertIn(rec["help"][:30], text)
        self.assertIn("Example: " + rec["example"], text)
        self.assertIn("Enter = next free name: ", text)
        self.assertIn("* = you must answer, Esc = go back", text)

    def test_question_mark_shows_more_help_but_is_a_letter_when_typing(self):
        item = ITEMS[2]
        form = M.Form(item, self.FOUND, TTY(), color=False, width=80)
        form.key("?")
        self.assertIn(mt.FIELD_TEXT[item["label"]]["init"]["more"][:30], "\n".join(form.lines()))
        form.key("down")                                                # any other key goes back to the short help
        self.assertNotIn(mt.FIELD_TEXT[item["label"]]["run"]["more"][:30], "\n".join(form.lines()))
        form.key("x")
        form.key("?")
        self.assertTrue(form.val[1].endswith("x?"))

    def test_plain_problems(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "run_chat3"))
            open(os.path.join(d, "run_chat3", "ckpt.npz"), "w").close()
            msg = M._check_new_run(os.path.join(d, "run_chat3"))
            self.assertIn("already exists", msg)
            self.assertIn("run_chat4", msg)
        item = ITEMS[2]
        form = M.Form(item, dict(runs=[], data=[], toks=[], torch=False), TTY(), color=False, width=80)
        form.val[0] = ""
        self.assertIn("is needed", form.problem(0))
        self.assertIn("Base model to copy", form.problem(0))

    def test_typed_line_questions_print_the_same_help_and_question_mark_helps(self):
        said, answers = [], iter(["?", "", "", "hello"])
        item = [i for i in ITEMS if i["cmd"] == "generate"][0]
        argv = M.collect(item, self.FOUND, ask=lambda q: next(answers), say=said.append)
        text = "\n".join(said)
        self.assertIn(mt.FIELD_TEXT[item["label"]]["ckpt"]["help"], text)
        self.assertIn("Example: " + mt.FIELD_TEXT[item["label"]]["ckpt"]["example"], text)
        self.assertIn("Enter = newest model: ", text)
        self.assertIn(mt.FIELD_TEXT[item["label"]]["ckpt"]["more"], text)
        self.assertEqual(argv[0], "generate")
        self.assertIn("hello", argv)


class Confirmations(unittest.TestCase):
    FOUND = dict(runs=[run("work3/run_pre", 50, 100)], data=["work3/data_pre"], toks=[], torch=True)
    ARGV = {
        "new": ["train", "--run", "run_new", "--data-dir", "data_chat", "--size", "s30m"],
        "continue": ["train", "--run", "work3/run_pre", "--data-dir", "work3/data_pre", "--size", "s30m"],
        "finetune": ["train", "--run", "run_chat4", "--data-dir", "data_chat", "--size", "s30m", "--init-from", "work3/run_pre/ckpt.npz"],
        "corpus": ["corpus", "--lang", "en", "--mode", "all"],
        "corpus_all": ["corpus", "--lang", "en,ceb", "--mode", "all"],
        "taglish": ["corpus", "--lang", "taglish", "--mode", "clean", "--mix"],
    }

    def lines(self, name):
        argv = self.ARGV[name]
        if argv[0] == "train":
            return M._train_lines(argv, self.FOUND)
        orig = M.importlib.util.find_spec
        M.importlib.util.find_spec = lambda n, *a: object() if n == "datasets" else orig(n, *a)
        try:
            return M._corpus_lines(argv)[0]
        finally:
            M.importlib.util.find_spec = orig

    def test_boxes_say_about_to_happen_created_not_changed_and_the_command_last(self):
        for name in self.ARGV:
            lines = [l for l in self.lines(name) if l]
            self.assertTrue(lines[0].startswith("You are about to"), name)
            text = "\n".join(lines)
            for want in ("What will happen:", "What will be created:", "What will NOT be changed:"):
                self.assertIn(want, text, name)
            self.assertTrue(lines[-1].startswith("Technical command (for experts): "), name)
            self.assertIn("persona_lm", lines[-1])

    def test_numbers_only_where_the_code_has_them(self):
        self.assertIn("unknown until it starts", "\n".join(self.lines("continue")))
        self.assertIn("Download size: about", "\n".join(self.lines("corpus")))
        self.assertIn("Download size: unknown", "\n".join(self.lines("corpus_all")))     # Cebuano has no fixed size
        self.assertIn("backup", "\n".join(self.lines("continue")))
        self.assertIn("base model", "\n".join(self.lines("finetune")))
        self.assertIn("run_pre", [l for l in self.lines("finetune") if "copy" in l][0])
        self.assertIn("public_mix", "\n".join(self.lines("taglish")))

    def test_enter_means_no_everywhere(self):
        for strict in (False, True):
            self.assertFalse(M.Confirm("t", ["x"], strict, False, TTY(), False, 80).run(iter(["enter"]).__next__))
            self.assertTrue(M.Confirm("t", ["x"], strict, False, TTY(), False, 80).run(iter(["y"]).__next__))
        self.assertFalse(M.confirm_train(self.ARGV["new"], self.FOUND, ask=lambda q: "", say=lambda *_: None))

    def test_confirm_screens_fit_60_and_80_columns_on_cp1252(self):
        for width in (60, 80):
            for name in self.ARGV:
                box = M.Confirm("Start?", self.lines(name), name == "taglish", False, StrictCp1252(), False, width, False)
                for line in box.lines():
                    line.encode("cp1252")
                    self.assertLessEqual(len(line), width - 1, (name, line))
                self.assertEqual(len({len(l) for l in box.lines()[1:]}), 1)


class SettingsScreens(unittest.TestCase):
    BASE = {"steps": 1000, "lr": 6e-4, "min_lr": 6e-5, "warmup": 100, "weight_decay": 0.1, "clip": 1.0, "batch": 2, "accum": 2, "eval_every": 200}

    def screen(self, answers, tokens=None, width=60):
        it, said = iter(list(answers) + [""] * 5), []
        changed = M.edit_settings(dict(self.BASE), 4096, 0, ask=lambda q: next(it), say=said.append, train_tokens=tokens, width=width)
        return changed, said

    def test_three_named_presets_in_the_asked_order(self):
        self.assertEqual([n for n, _ in mt.PRESET_TEXT], ["Safe (recommended)", "Gentle chat tuning", "Stop it memorizing"])
        changed, said = self.screen(["p", "2", ""])
        self.assertEqual(changed, {"lr": 1e-4, "min_lr": 1e-5, "warmup": 50, "steps": 3000})
        self.assertIn("Safe (recommended)", "\n".join(said))

    def test_screens_fit_60_columns_and_use_plain_names(self):
        for answers in ([""], ["1", ""], ["p", ""], ["5", "0.2", ""], ["5", "9", ""], ["2", "abc", ""], ["zzz", ""]):
            _, said = self.screen(answers)
            for line in said:
                self.assertLessEqual(len(line), 59, line)
        text = "\n".join(self.screen([""])[1])
        for d in M.SETTINGS:
            self.assertIn(d["name"], text)

    def test_memorizing_warning_in_words_when_the_data_size_is_known(self):
        text = "\n".join(self.screen([""], tokens=1_000_000)[1])         # 1000 steps x 16,384 tokens = about 16 passes
        flat = lambda t: " ".join(t.split())                            # the screen wraps lines
        self.assertIn("passes over your data", flat(text))
        self.assertIn("More than about 4 passes", flat(text))
        self.assertNotIn("More than about 4 passes", flat("\n".join(self.screen([""], tokens=100_000_000)[1])))

    def test_tokens_per_file_size_only(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "train.bin"), "wb") as fh:
                fh.write(b"\0" * 2000)
            self.assertEqual(M._train_tokens({"data": d}), 1000)
        self.assertIsNone(M._train_tokens({"data": os.path.join("no", "such")}))
        self.assertIsNone(M._train_tokens({}))

if __name__ == "__main__":
    unittest.main()
