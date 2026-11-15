"""Stage U4b: the cleaner on the Data page (clean-preview, repeats, clean-apply). Real loopback sockets, a temporary project, no browser."""
import filecmp
import json
import os
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import ui_server as UI
from test_ui_u0 import SECRET
from test_ui_u4a import DataBase

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ex(chat, session, target, role="assistant", ctx="hello there"):
    return {"chat": chat, "session": session, "messages": [{"role": "user", "content": ctx, "train": role == "user"}] + [{"role": role, "content": target, "train": True}]}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def jl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class CleanBase(DataBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        shutil.copy(os.path.join(REPO, "clean_chat.py"), os.path.join(cls.root, "clean_chat.py"))
        cls.write_import()
        cls.had_link = os.path.islink(os.path.join(cls.root, "data_msgr_chats"))

    @classmethod
    def write_import(cls):
        d = os.path.join(cls.root, "data_msgr")
        train = [ex("chat_001", i, "Kuyog ko") for i in range(1, 6)]                       # 5 identical short replies: 2 over the limit of 3
        train += [ex("chat_001", 6, SECRET), ex("chat_001", 7, SECRET), ex("chat_001", 8, "see you later at the market")]
        train += [ex("chat_001", 9, "You joined the call")]                                  # a system notice as the target: nothing left to learn
        train += [ex("chat_002", 1, "Person_3: hello there", role="user")]                    # another person's line in a group chat
        train += [ex("chat_002", 2, "okay sige mamaya")]
        jl(os.path.join(d, "train.jsonl"), train)
        jl(os.path.join(d, "val.jsonl"), [ex("chat_001", 20, "unya ra"), ex("chat_002", 20, "Person_4: ayaw", role="user")])

    def get(self, path, **kw):
        st, _, data = self.api("GET", path, **kw)
        return st, json.loads(data), data

    def post(self, path, body=None, **kw):
        st, _, data = self.api("POST", path, {} if body is None else body, **kw)
        return st, json.loads(data), data

    def free_out(self):
        """The test layout makes data_msgr_chats a link that leaves the project: take it away for the tests that write."""
        p = os.path.join(self.root, "data_msgr_chats")
        if os.path.islink(p):
            os.unlink(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)                                                                # left by an earlier test of this class
        return p


class Gate(CleanBase):
    def test_new_routes_need_token_and_host(self):
        for p in ("/api/data/clean-preview", "/api/data/repeats"):
            self.assertEqual(self.api("GET", p, token=False)[0], 401, p)
            self.assertEqual(self.api("GET", p, token="wrong")[0], 401, p)
            self.assertEqual(self.api("GET", p, host="evil.example.com")[0], 403, p)
            self.assertEqual(self.api("GET", p, headers={"Origin": "http://evil.example.com"})[0], 403, p)
        p = "/api/data/clean-apply"
        self.assertEqual(self.api("POST", p, {"plan_id": "x"}, token=False)[0], 401)
        self.assertEqual(self.api("POST", p, {"plan_id": "x"}, token="wrong")[0], 401)
        self.assertEqual(self.api("POST", p, {"plan_id": "x"}, host="evil.example.com")[0], 403)
        self.assertEqual(self.api("POST", p, "plan_id=x", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.api("GET", p)[0], 404)                                       # apply is POST only

    def test_routes_are_listed_exactly(self):
        self.assertIn("/api/data/clean-apply", UI.ROUTES["POST"])
        for p in ("/api/data/clean-preview", "/api/data/repeats"):
            self.assertIn(p, UI.ROUTES["GET"])
            self.assertNotIn(p, UI.ROUTES["POST"])


class Preview(CleanBase):
    def test_preview_counts(self):
        st, d, raw = self.get("/api/data/clean-preview")
        self.assertEqual(st, 200, d)
        t, v = d["splits"]["train"], d["splits"]["val"]
        self.assertEqual((t["before"], t["after"]), (11, 7))
        self.assertEqual(t["removed"]["dropped_repeated_short"], 2)
        self.assertEqual(t["removed"]["dropped_no_target"], 1)
        self.assertEqual(t["removed"]["dropped_other_group_speaker"], 1)
        self.assertEqual(t["removed"]["system_msgs_removed"], 1)
        self.assertEqual((v["before"], v["after"]), (2, 1))
        self.assertEqual(d["chats"]["before"], 2)
        self.assertEqual(d["chats"]["with_train_file"], 2)
        self.assertEqual(d["params"], {"max_dup": 3, "max_dup_long": 5, "keep_others": False})
        self.assertEqual(d["replies"]["before"]["total"], 10)                              # 11 examples, one is a user-role target
        self.assertEqual(d["replies"]["after"]["total"], 7)
        self.assertTrue(d["plan_id"])
        self.assertIn("python clean_chat.py --dir data_msgr --max-dup 3", d["command"])

    def test_preview_sends_no_text_and_writes_nothing(self):
        before = sorted(os.listdir(self.root))
        raw = self.get("/api/data/clean-preview")[2]
        self.assertNotIn(SECRET.encode(), raw)
        self.assertNotIn(b"Kuyog", raw)
        self.assertNotIn(b"market", raw)
        self.assertEqual(sorted(os.listdir(self.root)), before)

    def test_params_change_the_result_and_are_checked(self):
        d = self.get("/api/data/clean-preview?max_dup=0&max_dup_long=0")[1]
        self.assertEqual(d["splits"]["train"]["removed"]["dropped_repeated_short"], 0)      # 0 = no limit
        d = self.get("/api/data/clean-preview?keep_others=1")[1]
        self.assertEqual(d["splits"]["train"]["removed"]["dropped_other_group_speaker"], 0)
        for bad in ("max_dup=abc", "max_dup=-1", "max_dup=101", "max_dup_long=1.5"):
            self.assertEqual(self.get("/api/data/clean-preview?" + bad)[0], 400, bad)

    def test_no_import_is_404(self):
        p = os.path.join(self.root, "data_msgr", "train.jsonl")
        os.rename(p, p + ".off")
        try:
            self.assertEqual(self.get("/api/data/clean-preview")[0], 404)
        finally:
            os.rename(p + ".off", p)

    def test_damaged_file_is_a_clean_error(self):
        p = os.path.join(self.root, "data_msgr", "train.jsonl")
        os.rename(p, p + ".off")
        with open(p, "w") as f:
            f.write("{broken\n")
        try:
            st, d, raw = self.get("/api/data/clean-preview")
            self.assertEqual(st, 422)
            self.assertNotIn(b"Traceback", raw)
        finally:
            os.remove(p)
            os.rename(p + ".off", p)

    def test_old_clean_chat_is_refused_with_a_clear_message(self):
        p = os.path.join(self.root, "clean_chat.py")
        os.rename(p, p + ".off")
        with open(p, "w") as f:
            f.write("def main():\n    pass\n")
        try:
            st, d, _ = self.get("/api/data/clean-preview")
            self.assertEqual(st, 409)
            self.assertIn("older version", d["error"])
        finally:
            os.remove(p)
            os.rename(p + ".off", p)


class Repeats(CleanBase):
    def test_default_has_no_text(self):
        st, d, raw = self.get("/api/data/repeats")
        self.assertEqual(st, 200)
        self.assertFalse(d["text_shown"])
        self.assertNotIn("top", d)
        self.assertNotIn(b"Kuyog", raw)
        self.assertNotIn(SECRET.encode(), raw)
        self.assertEqual(d["before"]["buckets"][1], {"label": "4-10 times", "replies": 1, "examples": 5})

    def test_text_is_opt_in_cut_and_limited(self):
        st, d, raw = self.get("/api/data/repeats?text=1")
        self.assertEqual(st, 200, d)
        rows = {r["reply"]: r for r in d["top"]}
        self.assertEqual(rows["kuyog ko"], {"reply": "kuyog ko", "before": 5, "after": 3})
        self.assertNotIn(SECRET.encode(), raw)                                              # 33 characters: cut to 30 plus an ellipsis
        self.assertIn(SECRET[:30].encode(), raw)
        self.assertLessEqual(len(d["top"]), UD_ROWS)
        self.assertNotIn(b"market", raw)                                                    # appears once: not a repeated reply

    def test_text_refused_when_not_anonymised(self):
        self.put_stats(dict(self.stats, anonymized=False, safe_to_upload=False))
        try:
            st, d, raw = self.get("/api/data/repeats?text=1")
            self.assertEqual(st, 403)
            self.assertNotIn(b"Kuyog", raw)
            self.assertEqual(self.get("/api/data/repeats")[0], 200)                         # counts are still fine
        finally:
            self.put_stats(self.stats)


UD_ROWS = 15


class Apply(CleanBase):
    def preview_id(self, q=""):
        return self.get("/api/data/clean-preview" + q)[1]["plan_id"]

    def test_needs_a_known_plan(self):
        self.free_out()
        for body in ({}, {"plan_id": "nope"}, {"plan_id": 5}, {"plan_id": None}):
            self.assertEqual(self.post("/api/data/clean-apply", body)[0], 409, body)
        self.assertFalse(os.path.exists(os.path.join(self.root, "data_msgr_chats")))

    def test_writes_the_same_files_as_the_terminal_command(self):
        out = self.free_out()
        pid = self.preview_id()
        st, d, raw = self.post("/api/data/clean-apply", {"plan_id": pid})
        self.assertEqual(st, 200, d)
        self.assertEqual((d["written"], d["chat_files"], d["train_examples"], d["val_examples"], d["moved_old_to"]), ("data_msgr_chats", 2, 7, 1, None))
        self.assertNotIn(SECRET.encode(), raw)
        self.assertIn("prepare --kind chat", d["next_command"])
        ref = os.path.join(self.tmp, "terminal_out")
        r = subprocess.run([sys.executable, "clean_chat.py", "--dir", "data_msgr", "--out", ref], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        cmp = filecmp.dircmp(out, ref)
        self.assertEqual((cmp.left_only, cmp.right_only, cmp.diff_files), ([], [], []))
        for sub in ("train", "val"):
            c = filecmp.dircmp(os.path.join(out, sub), os.path.join(ref, sub))
            self.assertEqual((c.left_only, c.right_only, c.diff_files), ([], [], []), sub)
        self.assertEqual(self.post("/api/data/clean-apply", {"plan_id": pid})[0], 409)       # a plan is used once

    def test_old_output_is_moved_not_deleted(self):
        out = self.free_out()
        pid = self.preview_id("?max_dup=2")
        self.post("/api/data/clean-apply", {"plan_id": pid})
        first = read(os.path.join(out, "train", "chat_001.jsonl"))
        pid = self.preview_id("?max_dup=3")
        st, d, _ = self.post("/api/data/clean-apply", {"plan_id": pid})
        self.assertEqual(st, 200)
        self.assertTrue(d["moved_old_to"].startswith("_to_delete"))
        moved = os.path.join(self.root, "_to_delete", os.path.basename(d["moved_old_to"].replace("\\", "/")))
        self.assertEqual(read(os.path.join(moved, "train", "chat_001.jsonl")), first)
        self.assertNotEqual(read(os.path.join(out, "train", "chat_001.jsonl")), first)

    def test_changed_import_refuses_the_old_plan(self):
        self.free_out()
        pid = self.preview_id()
        p = os.path.join(self.root, "data_msgr", "train.jsonl")
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(ex("chat_003", 1, "a brand new reply")) + "\n")
        try:
            st, d, _ = self.post("/api/data/clean-apply", {"plan_id": pid})
            self.assertEqual(st, 409)
            self.assertFalse(os.path.exists(os.path.join(self.root, "data_msgr_chats", "train")))
        finally:
            self.write_import()

    def test_a_link_out_of_the_project_is_refused_and_nothing_is_written_outside(self):
        if not self.had_link:
            self.skipTest("no symlinks here")
        link = os.path.join(self.root, "data_msgr_chats")
        if not os.path.islink(link):
            os.symlink(self.outside, link)
        pid = self.preview_id()
        before = sorted(os.listdir(self.outside))
        st, d, raw = self.post("/api/data/clean-apply", {"plan_id": pid})
        self.assertEqual(st, 403, d)
        self.assertEqual(sorted(os.listdir(self.outside)), before)
        self.assertEqual(self.get("/api/data/clean-preview")[1]["output"]["usable"], False)

    def test_model_text_cannot_reach_it(self):
        self.free_out()
        st, d, _ = self.post("/api/data/clean-apply", {"plan_id": "/approve 1", "message": "CHANGE IDENTITY", "out": "../x", "dir": "work"})
        self.assertEqual(st, 409)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "x")))


if __name__ == "__main__":
    unittest.main()
