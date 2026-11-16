"""Stage 6: read-only data audit (duplicates, real vs synthetic, train/val leak)."""
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import cli, datatools as D  # noqa: E402

SECRET = "zebra-secret-phrase"


def ex(chat, session, reply, ctx="hello there my friend, how are you doing today"):
    return {"chat": chat, "session": session, "messages": [{"role": "user", "content": ctx, "train": False},
                                                          {"role": "assistant", "content": reply, "train": True}]}


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")


def rd(path, mode="r"):
    with open(path, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        return f.read()


def tree_hash(root):
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(root)):
        for n in sorted(files):
            h.update(n.encode())
            h.update(rd(os.path.join(d, n), "rb"))
    return h.hexdigest()


def project(tmp, leak=False, dup=False):
    tr, va = os.path.join(tmp, "data_clean", "train"), os.path.join(tmp, "data_clean", "val")
    real_tr = [ex("c%d" % (i % 4), i % 4, "my own reply number %d %s" % (i, SECRET), ctx="context %d is long enough to count here" % i) for i in range(20)]
    syn_tr = [ex("unknown", 0, "synthetic reply %d with some words in it" % i, ctx="synthetic context %d long enough to count" % i) for i in range(60)]
    if dup:
        real_tr += [real_tr[0], real_tr[0]]
        syn_tr += [ex("unknown", 0, "ok") for _ in range(9)]                       # a short reply repeated far above the limit
    real_va = [ex("c%d" % (i % 4), 10 + i, "different own reply %d long enough" % i, ctx="val context %d with enough characters" % i) for i in range(5)]
    if leak:
        real_va.append(real_tr[3])                                                   # same example on both sides
    write(os.path.join(tr, "merged_chats_cleanData.jsonl"), real_tr + ["not json"])
    write(os.path.join(tr, "synthetic_cleanData.jsonl"), syn_tr)
    write(os.path.join(va, "merged_chats_cleanData.jsonl"), real_va)
    write(os.path.join(va, "synthetic_cleanData.jsonl"), [ex("unknown", 0, "synthetic val reply %d with other words" % i, ctx="synthetic val context %d with characters" % i) for i in range(3)])
    return tr, va


class Audit(unittest.TestCase):
    def go(self, tmp, tr, va, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = D.audit(tr, va, os.path.join(tmp, "audit"), **kw)
        return code, buf.getvalue(), json.loads(rd(os.path.join(tmp, "audit", "report.json")))

    def test_clean_data_is_ok_and_ratio_is_right(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr, va = project(tmp)
            code, out, rep = self.go(tmp, tr, va, real_parts=["merged_chats"])
            self.assertEqual(code, 0, out)
            self.assertEqual(rep["leak_check"]["exact_examples_in_both"], 0)
            self.assertEqual(rep["leak_check"]["sessions_in_both"], 0)
            rs = rep["real_vs_synthetic"]
            self.assertEqual((rs["real_replies"], rs["synthetic_replies"]), (20, 60))
            self.assertEqual(rs["real_share"], 0.25)
            self.assertEqual(rep["dedup"]["exact_duplicate_examples"], 0)
            self.assertEqual(rep["val_files"][0]["examples"] + rep["val_files"][1]["examples"], 8)

    def test_leak_is_a_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr, va = project(tmp, leak=True)
            code, out, rep = self.go(tmp, tr, va)
            self.assertEqual(code, 1)
            self.assertEqual(rep["leak_check"]["exact_examples_in_both"], 1)
            self.assertIn("PROBLEM", out)

    def test_shared_session_is_a_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr, va = project(tmp)
            write(os.path.join(va, "merged_chats_cleanData.jsonl"), [ex("c1", 1, "a brand new reply with enough letters", ctx="unseen context with enough letters here"), ex("c2", 50, "another unseen reply with letters")])
            code, out, rep = self.go(tmp, tr, va)
            self.assertEqual(rep["leak_check"]["sessions_in_both"], 1)
            self.assertEqual(code, 1)

    def test_duplicates_and_stock_replies_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr, va = project(tmp, dup=True)
            code, out, rep = self.go(tmp, tr, va)
            self.assertEqual(rep["dedup"]["exact_duplicate_examples"], 2 + 8)        # 2 copies of one example + 8 repeats of "ok"
            self.assertEqual(rep["dedup"]["replies_over_clean_data_limit"], 1)
            tsv = rd(os.path.join(tmp, "audit", "exact_duplicates.tsv")).splitlines()
            self.assertEqual(len(tsv), 1 + 10)
            self.assertEqual(code, 0)                                                 # warnings, not a leak

    def test_never_writes_chat_text_and_never_changes_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr, va = project(tmp, dup=True)
            before = tree_hash(os.path.join(tmp, "data_clean"))
            code, out, rep = self.go(tmp, tr, va, real_parts=["merged"])
            self.assertEqual(before, tree_hash(os.path.join(tmp, "data_clean")))
            self.assertNotIn(SECRET, out)
            for n in os.listdir(os.path.join(tmp, "audit")):
                text = rd(os.path.join(tmp, "audit", n))
                self.assertNotIn(SECRET, text)
                self.assertNotIn("hello there my friend", text)

    def test_out_folder_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr, va = project(tmp)
            with self.assertRaises(ValueError):
                D.audit(tr, va, os.path.join(tr, "report"), say=lambda *_: None)         # inside the data
            os.makedirs(os.path.join(tmp, "audit"))
            write(os.path.join(tmp, "audit", "x.txt"), ["keep"])
            with self.assertRaises(ValueError):
                D.audit(tr, va, os.path.join(tmp, "audit"), say=lambda *_: None)         # not empty: nothing overwritten
            self.assertEqual(rd(os.path.join(tmp, "audit", "x.txt")).strip(), "keep")
            with self.assertRaises(FileNotFoundError):
                D.audit(os.path.join(tmp, "nope"), va, os.path.join(tmp, "o2"), say=lambda *_: None)

    def test_unmatched_real_part_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr, va = project(tmp)
            code, out, rep = self.go(tmp, tr, va, real_parts=["nothing_like_this"])
            self.assertEqual(rep["real_vs_synthetic"]["unmatched_parts"], ["nothing_like_this"])
            a = cli.build_parser().parse_args(["data-audit", "--train", tr, "--val", va, "--out", os.path.join(tmp, "o3"), "--real", "merged", "syn"])
            self.assertEqual(a.real, ["merged", "syn"])
            with self.assertRaises(SystemExit) as cm:
                cli.main(["data-audit", "--train", os.path.join(tmp, "missing"), "--val", va, "--out", os.path.join(tmp, "o4")])
            self.assertIn("error:", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
