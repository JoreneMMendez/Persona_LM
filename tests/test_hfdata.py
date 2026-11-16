"""Dataset recipes (persona_lm/hfdata.py). No network: a fake opener plays the part of the streamed dataset."""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from persona_lm import hfdata as H
from persona_lm import menu


def slurp(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def row(msg, score=3.0, en=0.99, donation=False, stream="s1", author="someone"):
    return {"message": msg, "score": score, "languages": [en, 0.01, 0.0], "donation": donation, "stream": {"id": stream}, "author": author}


def opener_for(rows):
    calls = []

    def op(recipe, skip=0):
        calls.append(skip)
        return iter(rows[skip:])
    op.calls = calls
    return op


def recipe(**kw):
    base = {"id": "t", "hf_repo": "owner/name", "text_field": "message", "max_kept": 1000,
            "filters": [{"field": "languages[0]", "op": ">", "value": 0.9}, {"field": "score", "op": ">", "value": 2.0}, {"field": "donation", "op": "==", "value": False}]}
    base.update(kw)
    return H.check_recipe(base)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.out = []
        self.say = self.out.append
        old = sys.stdout
        sys.stdout = io.StringIO()                                   # the plain progress lines are not interesting here
        self.addCleanup(setattr, sys, "stdout", old)


class TestRecipeChecks(unittest.TestCase):
    def test_good_recipe_is_completed(self):
        r = recipe()
        self.assertEqual(r["split"], "train")
        self.assertEqual(r["clean"]["min_chars"], 4)
        self.assertTrue(r["private_only"])

    def test_bad_repo_ids(self):
        for bad in ("https://huggingface.co/datasets/a/b", "a", "a/b/c", "../x/y", "a b/c", "", None):
            with self.assertRaises(H.HFError, msg=repr(bad)):
                recipe(hf_repo=bad)

    def test_bad_filters(self):
        for f in ({"field": "score", "op": "~", "value": 1}, {"field": "score", "op": ">", "value": "high"}, {"field": "__import__('os')", "op": "==", "value": 1},
                  {"field": "a b", "op": "==", "value": 1}, {"field": "x", "op": "in", "value": 3}, {"field": "x", "op": "==", "value": {"a": 1}}, "score > 1"):
            with self.assertRaises(H.HFError, msg=repr(f)):
                recipe(filters=[f])

    def test_unknown_cleaning_option_and_bad_id(self):
        with self.assertRaises(H.HFError):
            recipe(clean={"min_charz": 3})
        with self.assertRaises(H.HFError):
            recipe(id="Bad Id")
        with self.assertRaises(H.HFError):
            recipe(group={"by": "stream.id", "lines": 0})

    def test_save_never_overwrites(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        p = H.save_recipe(recipe(), d)
        self.assertTrue(os.path.exists(p))
        with self.assertRaises(H.HFError):
            H.save_recipe(recipe(), d)
        self.assertEqual(H.load_recipe("t", d)["hf_repo"], "owner/name")
        self.assertEqual(H.list_recipe_ids(d), ["t"])


class TestFilters(unittest.TestCase):
    def test_path_and_ops(self):
        r = row("hello", score=5, en=0.95)
        t = lambda f, op, v: H.test_filter({"field": f, "op": op, "value": v}, r)
        self.assertTrue(t("languages[0]", ">", 0.9))
        self.assertFalse(t("languages[0]", ">", 0.99))
        self.assertTrue(t("stream.id", "==", "s1"))
        self.assertTrue(t("stream.id", "in", ["s1", "s2"]))
        self.assertTrue(t("message", "contains", "ell"))
        self.assertTrue(t("donation", "==", False))
        self.assertFalse(t("donation", "==", 0))                 # False is not 0
        self.assertFalse(t("nothing.here", "==", 1))             # a missing field fails
        self.assertFalse(t("languages[9]", ">", 0))
        self.assertFalse(t("message", ">", 1))                   # wrong type fails

    def test_first_failing_is_counted_in_order(self):
        r = recipe()
        self.assertIsNone(H.first_failing(r["filters"], row("hello there")))
        self.assertEqual(H.first_failing(r["filters"], row("hello", en=0.2)), 0)
        self.assertEqual(H.first_failing(r["filters"], row("hello", score=1)), 1)
        self.assertEqual(H.first_failing(r["filters"], row("hello", donation=True)), 2)


class TestCleaner(unittest.TestCase):
    def setUp(self):
        self.c = H.Cleaner(recipe()["clean"])

    def test_tags_are_replaced_or_dropped(self):
        t, _ = self.c.clean("hi <|liver:text=Ymlib28=|> you are great <|:bijouSmile:|>")
        self.assertEqual(t, "hi [streamer] you are great")
        t, _ = self.c.clean("<|fans:text=cGViYmxl|>s are here")
        self.assertEqual(t, "[fans]s are here")
        t, _ = self.c.clean("<|maybe-collaborator:liver=Rin Penrose:text=cmlu:affiliation=none|> joined")
        self.assertEqual(t, "[streamer] joined")
        self.assertNotIn("<|", t)

    def test_emote_word_and_keep(self):
        t, _ = H.Cleaner(recipe(clean={"emotes": "word"})["clean"]).clean("nice <|:bijouSmile:|>")
        self.assertEqual(t, "nice bijouSmile")
        t, _ = H.Cleaner(recipe(clean={"emotes": "keep"})["clean"]).clean("nice <|:bijouSmile:|>")
        self.assertEqual(t, "nice :bijouSmile:")

    def test_spam_is_collapsed(self):
        self.assertEqual(self.c.clean("sooooooooo good")[0], "sooo good")
        self.assertEqual(self.c.clean("hahahahahahaha")[0], "hahaha")
        self.assertEqual(self.c.clean("pog pog pog pog pog pog")[0], "pog pog pog")
        self.assertEqual(self.c.clean("wow!!!!!!!!!!")[0], "wow!!!")
        self.assertEqual(self.c.clean("a normal sentence")[0], "a normal sentence")

    def test_short_and_placeholder_only_messages_are_dropped(self):
        self.assertEqual(self.c.clean("ok"), (None, "too short"))
        self.assertEqual(self.c.clean("<|liver:text=abc|>"), (None, "too short"))
        self.assertEqual(self.c.clean("   "), (None, "too short"))

    def test_links_and_handles_are_scrubbed(self):
        t, _ = self.c.clean("see https://example.com/x and mail me a@b.com please")
        self.assertNotIn("example.com", t)
        self.assertNotIn("a@b.com", t)


class TestScan(Tmp):
    def rows(self, n=40):
        out = []
        for i in range(n):
            out.append(row("message number %d is here" % i, stream="s%d" % (i // 10)))
        out.append(row("message number 3 is here"))                  # duplicate
        out.append(row("english but low score", score=0.5))
        out.append(row("not english at all", en=0.1))
        out.append(row("a donation message", donation=True))
        out.append(row("ok"))
        return out

    def test_scan_and_assemble(self):
        r = recipe(group={"by": "stream.id", "lines": 4})
        st = H.scan(r, self.d, opener_for(self.rows()), say=self.say)
        self.assertEqual(st["ended"], "end")
        self.assertEqual(st["kept"], 40)
        self.assertEqual(st["rows_scanned"], 45)
        self.assertEqual(st["drops"]["duplicate"], 1)
        self.assertEqual(st["drops"]["too short"], 1)
        info = H.assemble(r, self.d, say=self.say)
        self.assertEqual(info["messages"], 40)
        self.assertEqual(info["documents"], 12)                      # 4 streams of 10 messages, at most 4 per document: 3 documents each
        docs = [json.loads(x)["text"] for x in slurp(info["path"]).splitlines()]
        self.assertTrue(all(len(d.split("\n")) <= 4 for d in docs))
        chat = slurp(os.path.join(self.d, "public_clean", "t", "chat_clean.txt")).splitlines()
        self.assertEqual(len(chat), 40)
        self.assertFalse(any(os.path.exists(os.path.join(self.d, "public_clean", "t", n)) for n in ("t.jsonl.part", "chat_clean.txt.part")))

    def test_author_is_never_written(self):
        r = recipe()
        H.scan(r, self.d, opener_for([row("hello there friend", author="SecretUser123")]), say=self.say)
        H.assemble(r, self.d, say=self.say)
        for base, _, names in os.walk(self.d):
            for n in names:
                self.assertNotIn("SecretUser123", slurp(os.path.join(base, n)))

    def test_no_grouping_gives_one_document_per_message(self):
        r = recipe()
        H.scan(r, self.d, opener_for(self.rows()), say=self.say)
        self.assertEqual(H.assemble(r, self.d, say=self.say)["documents"], 40)

    def test_cap_stops_the_scan(self):
        r = recipe(max_kept=7)
        st = H.scan(r, self.d, opener_for(self.rows()), say=self.say)
        self.assertEqual((st["ended"], st["kept"]), ("cap", 7))

    def test_resume_after_ctrl_c_keeps_everything_and_skips_what_was_read(self):
        rows = self.rows()

        def interrupted(recipe_, skip=0):
            def gen():
                for i, x in enumerate(rows[skip:]):
                    if skip == 0 and i == 12:
                        raise KeyboardInterrupt()
                    yield x
            return gen()

        r = recipe(shard_size=100)
        with self.assertRaises(KeyboardInterrupt):
            H.scan(r, self.d, interrupted, say=self.say)
        s = H._load_state(H.work_dir(r, self.d))
        self.assertEqual((s["rows_scanned"], s["kept"], s["ended"]), (12, 12, None))
        with self.assertRaises(H.HFError):                           # not finished: assemble refuses
            H.assemble(r, self.d, say=self.say)
        op = opener_for(rows)
        st = H.scan(r, self.d, op, say=self.say)
        self.assertEqual(op.calls, [12])                             # asked the source to skip the 12 rows already done
        self.assertEqual((st["ended"], st["kept"]), ("end", 40))
        self.assertEqual(H.assemble(r, self.d, say=self.say)["messages"], 40)

    def test_resume_keeps_dedup_across_the_restart(self):
        rows = [row("same message here")] + [row("other message %d here" % i) for i in range(5)] + [row("same message here")]

        def stop_at_4(recipe_, skip=0):
            def gen():
                for i, x in enumerate(rows[skip:]):
                    if skip == 0 and i == 4:
                        raise KeyboardInterrupt()
                    yield x
            return gen()
        r = recipe()
        with self.assertRaises(KeyboardInterrupt):
            H.scan(r, self.d, stop_at_4, say=self.say)
        st = H.scan(r, self.d, opener_for(rows), say=self.say)
        self.assertEqual(st["kept"], 6)                              # the repeat of the first message is still dropped after the restart

    def test_network_error_saves_progress_and_says_so(self):
        rows = self.rows()

        def flaky(recipe_, skip=0):
            def gen():
                for i, x in enumerate(rows[skip:]):
                    if skip == 0 and i == 10:
                        raise ConnectionError("wifi gone")
                    yield x
            return gen()
        r = recipe()
        with self.assertRaises(H.HFError) as cm:
            H.scan(r, self.d, flaky, say=self.say)
        self.assertIn("run the same command", str(cm.exception))
        self.assertEqual(H._load_state(H.work_dir(r, self.d))["kept"], 10)
        self.assertEqual(H.scan(r, self.d, opener_for(rows), say=self.say)["kept"], 40)

    def test_changed_filters_are_refused_until_restart_and_old_work_is_moved(self):
        r = recipe()
        H.scan(r, self.d, opener_for(self.rows()), say=self.say)
        r2 = recipe(filters=[{"field": "score", "op": ">", "value": 4.0}])
        with self.assertRaises(H.HFError):
            H.scan(r2, self.d, opener_for(self.rows()), say=self.say)
        st = H.scan(r2, self.d, opener_for(self.rows()), say=self.say, restart=True)
        self.assertEqual(st["kept"], 0)                              # score is 3.0 everywhere or lower: nothing passes
        self.assertTrue(os.path.isdir(os.path.join(self.d, "_to_delete")))
        self.assertEqual(len(os.listdir(os.path.join(self.d, "_to_delete"))), 1)

    def test_failed_assemble_leaves_the_old_output(self):
        r = recipe()
        H.scan(r, self.d, opener_for(self.rows()), say=self.say)
        H.assemble(r, self.d, say=self.say)
        p = os.path.join(self.d, "public_clean", "t", "t.jsonl")
        before = slurp(p)
        os.remove(os.path.join(H.work_dir(r, self.d), "shard_000001.jsonl"))
        with self.assertRaises(OSError):
            H.assemble(r, self.d, say=self.say)
        self.assertEqual(slurp(p), before)

    def test_already_scanned_does_not_read_the_dataset_again(self):
        r = recipe()
        H.scan(r, self.d, opener_for(self.rows()), say=self.say)

        def boom(*a, **k):
            raise AssertionError("must not open the dataset again")
        st = H.scan(r, self.d, boom, say=self.say)
        self.assertEqual(st["kept"], 40)


class TestPeek(unittest.TestCase):
    def test_peek_shows_quantiles_and_filter_effects(self):
        rows = [row("message number %d is here" % i, score=float(i % 10)) for i in range(200)] + [row("not english", en=0.1, score=9.0)] * 20
        r = recipe(filters=[{"field": "languages[0]", "op": ">", "value": 0.9}, {"field": "score", "op": ">", "value": 5.0}])
        info = H.analyze(r, rows)
        self.assertEqual(info["drops"][0], 20)
        q = info["quant"]["score"]
        self.assertEqual(q[0.5], 5.0)                                # median among the English rows only
        text = "\n".join(H.format_peek(r, info))
        self.assertIn("filter 1", text)
        self.assertIn("score among rows that pass the OTHER filters", text)
        self.assertIn("sample:", text)

    def test_describe_rows_hides_text_examples(self):
        fields = H.describe_rows([row("hello world", author="SecretUser123")] * 5)
        blob = json.dumps([{k: v for k, v in f.items() if k != "vals"} for f in fields])
        self.assertNotIn("SecretUser123", blob)
        self.assertNotIn("hello world", blob)
        self.assertEqual(H.guess_text_field(fields), "message")
        paths = {f["path"] for f in fields}
        self.assertTrue({"score", "languages[0]", "donation", "stream.id"} <= paths)


class TestWizard(Tmp):
    def test_wizard_saves_a_recipe_from_scripted_answers(self):
        rows = [row("message number %d is here" % i, score=float(i % 10), stream="s%d" % (i // 20)) for i in range(100)]
        answers = iter([
            "owner/data", "train", "",                                # repo, split, config
            "message",                                                # text field
            "languages[0]", ">", "0.9",                               # filter 1
            "score", ">", "2.5",                                      # filter 2
            "donation", "==", "false",                                # filter 3
            "",                                                       # done with filters
            "4", "3", "y", "drop",                                    # cleaning
            "stream.id", "25",                                        # grouping
            "1000",                                                   # max kept
            "", "My data", "CC-BY-4.0", "credit owner",               # id (default), label, licence, attribution
            "y"])                                                     # save
        rid = H.wizard(self.d, ask=lambda q: next(answers), say=self.say, opener=opener_for(rows), rows_n=100)
        self.assertEqual(rid, "data")
        r = H.load_recipe("data", self.d)
        self.assertEqual(r["hf_repo"], "owner/data")
        self.assertEqual([H.filter_label(f) for f in r["filters"]], ["languages[0] > 0.9", "score > 2.5", "donation == false"])
        self.assertEqual(r["group"], {"by": "stream.id", "lines": 25})
        self.assertEqual(r["license"], "CC-BY-4.0")
        self.assertTrue(any("Fields found" in line for line in self.out))
        self.assertFalse(any("message number" in line and "Fields" in line for line in self.out))

    def test_wizard_refuses_a_link_then_accepts_a_name_and_can_be_backed_out(self):
        rows = [row("message number %d is here" % i) for i in range(10)]
        answers = iter(["https://huggingface.co/datasets/a/b", "a/b", "train", "", "message", "", "4", "3", "y", "drop", "", "100", "", "x", "", "", "n"])
        rid = H.wizard(self.d, ask=lambda q: next(answers), say=self.say, opener=opener_for(rows), rows_n=10)
        self.assertIsNone(rid)                                       # answered "n" to Save
        self.assertTrue(any("owner/name" in line for line in self.out))
        self.assertEqual(H.list_recipe_ids(self.d), [])

        def ctrl_c(q):
            raise KeyboardInterrupt()
        self.assertIsNone(H.wizard(self.d, ask=ctrl_c, say=self.say))


class TestCommands(Tmp):
    def parse(self, *argv):
        p = argparse.ArgumentParser()
        H.add_arguments(p)
        return p.parse_args(list(argv))

    def test_list_show_peek_run_and_no_start_without_yes(self):
        H.save_recipe(recipe(), self.d)
        rows = [row("message number %d is here" % i) for i in range(30)]
        H.cmd(self.parse("list"), self.d, say=self.say)
        self.assertTrue(any("owner/name" in x for x in self.out))
        H.cmd(self.parse("show", "t"), self.d, say=self.say)
        H.cmd(self.parse("peek", "--rows", "30"), self.d, say=self.say, opener=opener_for(rows))
        self.assertTrue(any("rows passing every filter" in x for x in self.out))
        H.cmd(self.parse("run", "t"), self.d, ask=lambda q: "n", say=self.say, opener=opener_for(rows))
        self.assertFalse(os.path.exists(H.work_dir(recipe(), self.d)))
        H.cmd(self.parse("run", "t", "--yes", "--plain"), self.d, say=self.say, opener=opener_for(rows))
        self.assertTrue(os.path.exists(os.path.join(self.d, "public_clean", "t", "t.jsonl")))
        self.assertIn("scanned", "\n".join(H.list_recipe_ids(self.d)) + H.status(recipe(), self.d))

    def test_errors_become_plain_messages(self):
        with self.assertRaises(SystemExit) as cm:
            H.cmd(self.parse("run", "nothing"), self.d, say=self.say)
        self.assertIn("no recipe", str(cm.exception))
        with self.assertRaises(SystemExit):
            H.cmd(self.parse("run"), self.d, say=self.say)           # which recipe?

    def test_the_cli_knows_the_command_and_the_menu_builds_a_valid_line(self):
        from persona_lm import cli
        p = cli.build_parser() if hasattr(cli, "build_parser") else None
        item = [i for i in menu.ITEMS if i["cmd"] == "hfdata"][0]
        argv = menu.build_argv(item, {"hf_action": "run", "hf_recipe": "oshichats"})
        self.assertEqual(argv, ["hfdata", "run", "oshichats"])
        self.assertEqual(menu.build_argv(item, {"hf_action": "new", "hf_recipe": ""}), ["hfdata", "new"])
        a = self.parse(*argv[1:])
        self.assertEqual((a.action, a.recipe), ("run", "oshichats"))
        self.assertIsNone(menu._check_recipe_id("good-id_1"))
        self.assertIsNotNone(menu._check_recipe_id("Bad Id"))
        if p is not None:
            self.assertEqual(p.parse_args(argv).action, "run")


class TestProgress(unittest.TestCase):
    def test_plain_mode_prints_lines_and_live_mode_redraws_one_line(self):
        t = [0.0]
        buf = io.StringIO()
        p = H.Progress(100, live=False, stream=buf, clock=lambda: t[0])
        p.update(10, 5)
        self.assertEqual(buf.getvalue(), "")                         # too soon for a plain line
        t[0] = 30.0
        p.update(1000, 50)
        self.assertIn("kept 50 / 100", buf.getvalue())
        self.assertIn("ETA", buf.getvalue())
        live = io.StringIO()
        q = H.Progress(100, live=True, stream=live, clock=lambda: t[0])
        t[0] = 31.0
        q.update(10, 5)
        q.log("hello")
        q.close()
        self.assertIn("\x1b[2K", live.getvalue())
        self.assertIn("hello", live.getvalue())


class TestKitAndTokens(Tmp):
    def test_colab_kit_contains_recipe_package_notebook_and_the_right_id(self):
        import zipfile
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "colab_hfdata.ipynb"), self.d)
        H.save_recipe(recipe(id="mydata"), self.d)
        z = H.colab_kit(H.load_recipe("mydata", self.d), self.d, say=self.say)
        names = zipfile.ZipFile(z).namelist()
        self.assertIn("persona/dataset_recipes/mydata.json", names)
        self.assertIn("persona/persona_lm/hfdata.py", names)
        self.assertIn("persona/colab_hfdata.ipynb", names)
        self.assertFalse(any("__pycache__" in n for n in names))
        nb = json.loads(zipfile.ZipFile(z).read("persona/colab_hfdata.ipynb"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn('RECIPE_ID = "mydata"', src)
        z2 = H.colab_kit(H.load_recipe("mydata", self.d), self.d, say=self.say)
        self.assertNotEqual(z, z2)                                   # never overwrites an earlier kit

    def test_tokens_needs_the_output_first_and_skips_a_finished_folder(self):
        r = recipe()
        with self.assertRaises(H.HFError):
            H.make_tokens(r, self.d, say=self.say)
        os.makedirs(os.path.join(self.d, "data_pre_t"))
        with open(os.path.join(self.d, "data_pre_t", "meta.json"), "w") as f:
            f.write("{}")
        os.makedirs(os.path.join(self.d, "public_clean", "t"))
        with open(os.path.join(self.d, "public_clean", "t", "t.jsonl"), "w") as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
        tok = os.path.join(self.d, "tok.json")
        with open(tok, "w") as f:
            f.write("{}")
        out = H.make_tokens(r, self.d, tokenizer=tok, say=self.say)
        self.assertTrue(out.endswith("data_pre_t"))
        self.assertTrue(any("already built" in x for x in self.out))


if __name__ == "__main__":
    unittest.main()
