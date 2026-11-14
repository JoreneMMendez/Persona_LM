"""Tests for clean_chat.py (one numbered file per chat; replaces the old merging cleaner)."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("clean_chat", os.path.join(ROOT, "clean_chat.py"))
CC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CC)


def ex(chat, session, target, role="assistant", before=(), mood=True):
    msgs = [{"role": "system", "content": "<|mood|> neutral +0.0 +0.0 <|end|>", "train": False}] if mood else []
    for r, c in before:
        msgs.append({"role": r, "content": c, "train": False})
    msgs.append({"role": role, "content": target, "train": True})
    return {"chat": chat, "session": session, "messages": msgs}


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(*args):
    out = io.StringIO()
    old = sys.argv
    sys.argv = ["clean_chat.py", *args]
    try:
        with contextlib.redirect_stdout(out):
            CC.main()
    finally:
        sys.argv = old
    return out.getvalue()


class PerChatFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = self.tmp.name
        self.out = os.path.join(self.d, "out")

    def p(self, name):
        return os.path.join(self.d, name)

    def test_one_numbered_file_per_chat_and_index(self):
        rows = [ex("chat_005", 1, "hello a"), ex("chat_002", 1, "hello b"), ex("chat_002", 2, "hello c"), ex("chat_011", 1, "hello d")]
        write(self.p("a.jsonl"), rows)
        run("--input", self.p("a.jsonl"), "--out", self.out)
        files = sorted(os.listdir(os.path.join(self.out, "train")))
        self.assertEqual(files, ["chat_001.jsonl", "chat_002.jsonl", "chat_003.jsonl"])
        self.assertEqual(len(read(os.path.join(self.out, "train", "chat_001.jsonl"))), 2)   # chat_002 has 2 examples, sorted first
        idx = json.load(open(os.path.join(self.out, "index.json"), encoding="utf-8"))
        self.assertEqual([f["original_chat"] for f in idx["files"]], ["chat_002", "chat_005", "chat_011"])
        self.assertEqual(idx["files"][0]["train_examples"], 2)

    def test_files_keep_input_order_across_inputs(self):
        write(self.p("a.jsonl"), [ex("chat_009", 1, "one")])
        write(self.p("b.jsonl"), [ex("syn_001", 1, "two")])
        run("--input", self.p("a.jsonl"), self.p("b.jsonl"), "--out", self.out)
        idx = json.load(open(os.path.join(self.out, "index.json"), encoding="utf-8"))
        self.assertEqual([f["original_chat"] for f in idx["files"]], ["chat_009", "syn_001"])

    def test_validation_is_split_by_whole_sessions(self):
        rows = [ex("chat_001", s, "reply %d-%d" % (s, k)) for s in range(1, 21) for k in range(2)]
        write(self.p("a.jsonl"), rows)
        run("--input", self.p("a.jsonl"), "--out", self.out, "--val-fraction", "0.1")
        tr = read(os.path.join(self.out, "train", "chat_001.jsonl"))
        va = read(os.path.join(self.out, "val", "chat_001.jsonl"))
        self.assertEqual({e["session"] for e in va}, {19, 20})
        self.assertTrue({e["session"] for e in tr}.isdisjoint({e["session"] for e in va}))
        self.assertEqual(len(tr) + len(va), 40)

    def test_small_chats_get_no_val_file(self):
        write(self.p("a.jsonl"), [ex("chat_001", s, "r%d" % s) for s in range(1, 4)])
        run("--input", self.p("a.jsonl"), "--out", self.out)
        self.assertFalse(os.path.exists(os.path.join(self.out, "val", "chat_001.jsonl")))

    def test_system_notice_removed_mood_line_kept(self):
        row = ex("chat_001", 1, "ok", before=[("user", "Person_01 joined the video call."), ("user", "hi there")])
        write(self.p("a.jsonl"), [row])
        run("--input", self.p("a.jsonl"), "--out", self.out)
        got = read(os.path.join(self.out, "train", "chat_001.jsonl"))[0]["messages"]
        self.assertEqual([m["role"] for m in got], ["system", "user", "assistant"])
        self.assertIn("<|mood|>", got[0]["content"])

    def test_target_that_is_a_system_notice_is_dropped(self):
        write(self.p("a.jsonl"), [ex("chat_001", 1, "You joined the call."), ex("chat_002", 1, "fine")])
        run("--input", self.p("a.jsonl"), "--out", self.out)
        self.assertEqual(os.listdir(os.path.join(self.out, "train")), ["chat_001.jsonl"])      # chat_001 had nothing left: no file
        idx = json.load(open(os.path.join(self.out, "index.json"), encoding="utf-8"))
        self.assertEqual(idx["files"][0]["original_chat"], "chat_002")

    def test_other_speaker_targets(self):
        write(self.p("a.jsonl"), [ex("chat_001", 1, "Person_65: gago", role="user"), ex("chat_001", 2, "mine")])
        run("--input", self.p("a.jsonl"), "--out", self.out)
        self.assertEqual(len(read(os.path.join(self.out, "train", "chat_001.jsonl"))), 1)
        run("--input", self.p("a.jsonl"), "--out", self.out, "--keep-others")
        self.assertEqual(len(read(os.path.join(self.out, "train", "chat_001.jsonl"))), 2)

    def test_repeat_caps_count_over_all_chats(self):
        rows = [ex("chat_00%d" % c, s, "Oo") for c in (1, 2) for s in range(1, 4)]            # 6 x "Oo"
        rows += [ex("chat_003", s, "a long repeated stock reply here") for s in range(1, 9)]    # 8 x long
        write(self.p("a.jsonl"), rows)
        run("--input", self.p("a.jsonl"), "--out", self.out, "--max-dup", "3", "--max-dup-long", "5")
        kept = [e["messages"][-1]["content"] for f in os.listdir(os.path.join(self.out, "train"))
                for e in read(os.path.join(self.out, "train", f))]
        self.assertEqual(kept.count("Oo"), 3)
        self.assertEqual(kept.count("a long repeated stock reply here"), 5)

    def test_zero_means_no_cap(self):
        write(self.p("a.jsonl"), [ex("chat_001", s, "Oo") for s in range(1, 8)])
        run("--input", self.p("a.jsonl"), "--out", self.out, "--max-dup", "0")
        self.assertEqual(len(read(os.path.join(self.out, "train", "chat_001.jsonl"))), 7)

    def test_rerun_removes_files_of_an_earlier_run(self):
        write(self.p("a.jsonl"), [ex("chat_001", 1, "a"), ex("chat_002", 1, "b")])
        run("--input", self.p("a.jsonl"), "--out", self.out)
        write(self.p("a.jsonl"), [ex("chat_001", 1, "a")])
        run("--input", self.p("a.jsonl"), "--out", self.out)
        self.assertEqual(os.listdir(os.path.join(self.out, "train")), ["chat_001.jsonl"])

    def test_dir_mode_uses_the_given_val_file(self):
        os.makedirs(self.p("data_msgr"))
        write(self.p("data_msgr/train.jsonl"), [ex("chat_001", 1, "t1"), ex("chat_001", 2, "t2")])
        write(self.p("data_msgr/val.jsonl"), [ex("chat_001", 9, "v1")])
        run("--dir", self.p("data_msgr"), "--out", self.out)
        self.assertEqual(len(read(os.path.join(self.out, "train", "chat_001.jsonl"))), 2)
        self.assertEqual(read(os.path.join(self.out, "val", "chat_001.jsonl"))[0]["messages"][-1]["content"], "v1")

    def test_missing_input_stops_with_a_message(self):
        with self.assertRaises(SystemExit):
            run("--input", self.p("nope.jsonl"), "--out", self.out)

    def test_output_is_readable_by_prepare_folder_walk(self):
        """prepare walks folders and reads only .json/.jsonl: index.json lives outside train/ and val/."""
        write(self.p("a.jsonl"), [ex("chat_001", s, "r%d" % s) for s in range(1, 13)])
        run("--input", self.p("a.jsonl"), "--out", self.out)
        sys.path.insert(0, ROOT)
        from persona_lm import data as D
        names = [os.path.basename(f) for f in D._files([os.path.join(self.out, "train")])]
        self.assertEqual(names, ["chat_001.jsonl"])


if __name__ == "__main__":
    unittest.main()
