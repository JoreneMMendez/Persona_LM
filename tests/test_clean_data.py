import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import clean_data as CD  # noqa: E402


def ex(chat, session, user, reply):
    return {"chat": chat, "session": session, "messages": [
        {"role": "user", "content": user, "train": False}, {"role": "assistant", "content": reply, "train": True}]}


def write(path, rows, raw_lines=()):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        for line in raw_lines:
            f.write(line + "\n")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


class CleanData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.out = os.path.join(self.d, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def run_clean(self, files, *extra):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return CD.main(["--input"] + files + ["--out", self.out] + list(extra))

    def test_one_output_per_input_named_cleandata_and_not_merged(self):
        a, b = os.path.join(self.d, "a.jsonl"), os.path.join(self.d, "b.jsonl")
        write(a, [ex("c1", i, "hello %d" % i, "reply a %d" % i) for i in range(40)])
        write(b, [ex("c2", i, "yo %d" % i, "reply b %d" % i) for i in range(40)])
        self.run_clean([a, b])
        self.assertEqual(sorted(os.listdir(os.path.join(self.out, "train"))), ["a_cleanData.jsonl", "b_cleanData.jsonl"])
        self.assertTrue(all(e["chat"] == "c1" for e in read(os.path.join(self.out, "train", "a_cleanData.jsonl"))))
        self.assertTrue(os.path.exists(os.path.join(self.out, "val", "a_cleanData.jsonl")))

    def test_bad_lines_duplicates_and_empty_messages_are_dropped(self):
        a = os.path.join(self.d, "a.jsonl")
        rows = [ex("c1", 0, "same", "reply one"), ex("c1", 0, "same", "reply one"), {"nothing": 1},
                {"chat": "c1", "session": 1, "messages": [{"role": "user", "content": "  ", "train": False}, {"role": "assistant", "content": "ok then", "train": True}]}]
        write(a, rows, raw_lines=["{not json"])
        rep = self.run_clean([a], "--val-fraction", "0")
        r = rep[0]
        self.assertEqual(r["unreadable_lines"], 1)
        self.assertEqual(r["dropped_duplicate_example"], 1)
        self.assertEqual(r["empty_message"], 1)
        self.assertEqual(r["train_out"], 2)

    def test_repeated_replies_are_limited_over_all_files(self):
        a, b = os.path.join(self.d, "a.jsonl"), os.path.join(self.d, "b.jsonl")
        write(a, [ex("c1", i, "q%d" % i, "lol") for i in range(3)])
        write(b, [ex("c2", i, "w%d" % i, "lol") for i in range(3)])
        self.run_clean([a, b], "--val-fraction", "0", "--max-dup", "3")
        total = sum(len(read(os.path.join(self.out, "train", n))) for n in os.listdir(os.path.join(self.out, "train")))
        self.assertEqual(total, 3)

    def test_cleandata_files_are_skipped_and_old_output_replaced(self):
        a = os.path.join(self.d, "a.jsonl")
        write(a, [ex("c1", i, "q%d" % i, "answer number %d" % i) for i in range(5)])
        self.run_clean([a], "--val-fraction", "0")
        self.run_clean([a, os.path.join(self.out, "train")], "--val-fraction", "0")
        self.assertEqual(os.listdir(os.path.join(self.out, "train")), ["a_cleanData.jsonl"])
        self.assertEqual(len(read(os.path.join(self.out, "train", "a_cleanData.jsonl"))), 5)

    def test_max_examples_caps_each_file(self):
        a = os.path.join(self.d, "a.jsonl")
        write(a, [ex("c1", i, "q%d" % i, "answer number %d" % i) for i in range(50)])
        self.run_clean([a], "--val-fraction", "0", "--max-examples", "10")
        n = len(read(os.path.join(self.out, "train", "a_cleanData.jsonl")))
        n += len(read(os.path.join(self.out, "val", "a_cleanData.jsonl"))) if os.path.exists(os.path.join(self.out, "val", "a_cleanData.jsonl")) else 0
        self.assertEqual(n, 10)      # (a chat with 10 or more sessions always holds one session out for validation)


if __name__ == "__main__":
    unittest.main()
