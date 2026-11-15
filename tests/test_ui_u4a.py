"""Stage U4a: the Data page routes (persona_lm/ui_data.py). Read only; counts and verdicts, never message text."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from persona_lm import ui_data as UD
from persona_lm import ui_server as UI
from persona_lm.bpe import EOS, Tokenizer
from test_ui_u0 import SECRET, Base

CORPUS = "the cat sat on the mat. luna loves mango and the dog. hello how are you today fine thanks. " * 40


def write_tokens(folder, tok, kind="chat", n=6000, loss=3000, with_val=True):
    os.makedirs(folder, exist_ok=True)
    meta = {"kind": kind, "vocab_size": tok.vocab_size, "tokenizer_sha256": tok.fingerprint()}
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f)
    ids = np.full(n, 5, dtype=np.uint16)
    ids[::100] = EOS                                  # one example every 100 tokens
    ids.tofile(os.path.join(folder, "train.bin"))
    if with_val:
        ids[:n // 10].tofile(os.path.join(folder, "val.bin"))
    if kind == "chat":
        m = np.zeros(n, dtype=np.uint8)
        m[:loss] = 1
        m.tofile(os.path.join(folder, "train_mask.bin"))


class DataBase(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        r = cls.root
        cls.tok = Tokenizer.train([CORPUS], 300)
        cls.tok.save(os.path.join(r, "work", "tok.json"))
        os.makedirs(os.path.join(r, "data_msgr"), exist_ok=True)
        cls.stats = {"chats_imported": 4, "sessions": 30, "my_messages": 120, "my_turns": 100, "examples_train": 500, "examples_val": 20,
                     "period": ["2020-01-01", "2024-05-05"], "learn_from_others": True, "anonymized": True, "paranoid": False,
                     "anonymizer_counts": {"phone": 3, "name": 40}, "audit": {"leaks": {}, "suspicious": {}},
                     "dropped_message_kinds": {"photo": 9}, "safe_to_upload": True, "data_size_level": "SMALL", "data_size_verdict": "Small but fine."}
        cls.put_stats(cls.stats)
        for n in ("train.jsonl", "val.jsonl"):
            with open(os.path.join(r, "data_msgr", n), "w") as f:
                f.write(json.dumps({"messages": [{"content": SECRET}]}) + "\n")
        with open(os.path.join(r, "data_msgr", "private", "audit.txt"), "w") as f:
            f.write("LEAK phone: " + SECRET)
        write_tokens(os.path.join(r, "data_chat4096"), cls.tok)

    @classmethod
    def put_stats(cls, d):
        with open(os.path.join(cls.root, "data_msgr", "stats.json"), "w") as f:
            json.dump(d, f)

    def get(self, path, **kw):
        st, _, data = self.api("GET", path, **kw)
        return st, json.loads(data), data


class Gate(DataBase):
    def test_needs_token_and_host(self):
        for p in ("/api/data/overview", "/api/data/audit", "/api/data/report"):
            self.assertEqual(self.api("GET", p, token=False)[0], 401, p)
            self.assertEqual(self.api("GET", p, token="wrong")[0], 401, p)
            self.assertEqual(self.api("GET", p, host="evil.example.com")[0], 403, p)
            self.assertEqual(self.api("GET", p, headers={"Origin": "http://evil.example.com"})[0], 403, p)

    def test_no_post_route_and_no_write_routes(self):
        for p in ("/api/data/overview", "/api/data/audit", "/api/data/report", "/api/data/import", "/api/data/scan", "/api/data/run"):
            self.assertEqual(self.api("POST", p, {"cmd": "del *", "path": "../x"})[0], 404, p)
        self.assertEqual([p for p in UI.ROUTES["POST"] if "/data" in p], ["/api/data/clean-apply"])       # U4b: the only data write, see test_ui_u4b.py

    def test_routes_are_listed(self):
        for p in ("/api/data/overview", "/api/data/audit", "/api/data/report"):
            self.assertIn(p, UI.ROUTES["GET"])
        self.assertIn("/static/data.js", UI.ROUTES["GET"])


class Overview(DataBase):
    def test_overview_steps_and_folders(self):
        st, d, raw = self.get("/api/data/overview")
        self.assertEqual(st, 200)
        self.assertEqual({s["id"] for s in d["steps"]}, {"scan", "import", "audit", "prepare", "report"})
        by = {s["id"]: s for s in d["steps"]}
        self.assertEqual(by["import"]["state"], "done")
        self.assertEqual(by["prepare"]["state"], "done")                         # data_chat4096 holds chat tokens
        self.assertEqual(by["scan"]["state"], "next")                            # no settings file: not remembered
        self.assertFalse(d["export_remembered"])
        self.assertTrue(by["scan"]["command"].startswith("python -m persona_lm messenger-scan"))
        self.assertIn("s30m", d["sizes"])
        names = {t["name"]: t for t in d["token_folders"]}
        self.assertTrue(names["data_chat4096"]["exists"])
        self.assertEqual(names["data_chat4096"]["train_tokens"], 6000)
        self.assertFalse(names["data_pre"]["exists"])
        self.assertNotIn(SECRET.encode(), raw)

    def test_export_path_is_never_sent(self):
        path = os.path.join(self.root, "persona_settings.json")
        with open(path, "w") as f:
            json.dump({"export": "C:\\Users\\someone\\facebook-secretname-export", "me": ["Secret Person"]}, f)
        try:
            st, d, raw = self.get("/api/data/overview")
            self.assertTrue(d["export_remembered"])
            self.assertNotIn(b"secretname", raw)
            self.assertNotIn(b"Secret Person", raw)
            self.assertEqual({s["id"]: s["state"] for s in d["steps"]}["scan"], "done")
        finally:
            os.remove(path)

    def test_damaged_stats_does_not_break_overview(self):
        with open(os.path.join(self.root, "data_msgr", "stats.json"), "w") as f:
            f.write("{not json")
        try:
            self.assertEqual(self.get("/api/data/overview")[0], 200)
            self.assertEqual(self.get("/api/data/audit")[0], 422)
        finally:
            self.put_stats(self.stats)


class Audit(DataBase):
    def test_clean_audit(self):
        st, d, raw = self.get("/api/data/audit")
        self.assertEqual(st, 200)
        self.assertTrue(d["exists"])
        self.assertEqual((d["level"], d["safe_to_upload"], d["chats"], d["my_messages"]), ("ok", True, 4, 120))
        self.assertEqual(d["masked"], {"phone": 3, "name": 40})
        self.assertEqual(d["dropped_message_kinds"], {"photo": 9})

    def test_never_shows_private_or_message_text(self):
        for p in ("/api/data/audit", "/api/data/overview", "/api/data/report?folder=data_chat4096&size=s10m"):
            self.assertNotIn(SECRET.encode(), self.get(p)[2], p)

    def test_leaks_are_bad_and_suspicious_is_a_warning(self):
        try:
            self.put_stats(dict(self.stats, audit={"leaks": {"phone": 2}, "suspicious": {}}, safe_to_upload=False))
            d = self.get("/api/data/audit")[1]
            self.assertEqual(d["level"], "bad")
            self.assertIn("2 possible leak", d["verdict"])
            self.put_stats(dict(self.stats, audit={"leaks": {}, "suspicious": {"name": 5}}))
            self.assertEqual(self.get("/api/data/audit")[1]["level"], "warn")
            self.put_stats(dict(self.stats, anonymized=False, safe_to_upload=False))
            d = self.get("/api/data/audit")[1]
            self.assertEqual(d["level"], "bad")
            self.assertIn("OFF", d["verdict"])
        finally:
            self.put_stats(self.stats)

    def test_no_import_yet(self):
        os.rename(os.path.join(self.root, "data_msgr", "stats.json"), os.path.join(self.root, "data_msgr", "stats.off"))
        try:
            st, d, _ = self.get("/api/data/audit")
            self.assertEqual((st, d["exists"]), (200, False))
        finally:
            os.rename(os.path.join(self.root, "data_msgr", "stats.off"), os.path.join(self.root, "data_msgr", "stats.json"))

    def test_only_fixed_folder_names(self):
        for bad in ("data_msgr/private", "../outside", "data_msgr_chats", "C:\\x", "work"):
            st, d, raw = self.get("/api/data/audit?folder=" + bad)
            self.assertEqual(st, 400, bad)
            self.assertNotIn(SECRET.encode(), raw)


class Report(DataBase):
    def test_chat_verdict_and_tokenizer_ok(self):
        st, d, _ = self.get("/api/data/report?folder=data_chat4096&size=s10m")
        self.assertEqual(st, 200, d)
        self.assertEqual((d["kind"], d["train_tokens"], d["val_tokens"], d["loss_tokens"], d["examples"]), ("chat", 6000, 600, 3000, 60))
        self.assertEqual(d["level"], "bad")                                         # 3,000 loss tokens is under 50k
        self.assertIn("VERY LITTLE", d["verdict"])
        self.assertEqual(d["tokenizer"]["state"], "ok")
        self.assertEqual(d["passes"]["steps_for_3_passes"], max(0, int(round(3 * 6000 / d["passes"]["tokens_per_step"]))))
        self.assertTrue(any("Validation data" in w for w in d["warnings"]))

    def test_other_tokenizer_is_flagged(self):
        other = Tokenizer.train(["completely different text here for another vocabulary. " * 50], 300)
        folder = os.path.join(self.root, "data_chat4096")
        write_tokens(folder, other)
        try:
            d = self.get("/api/data/report?folder=data_chat4096&size=s10m")[1]
            self.assertEqual(d["tokenizer"]["state"], "bad")
        finally:
            write_tokens(folder, self.tok)

    def test_missing_folder_and_bad_size(self):
        self.assertEqual(self.get("/api/data/report?folder=data_pre&size=s10m")[0], 404)
        self.assertEqual(self.get("/api/data/report?folder=data_chat4096&size=huge")[0], 400)
        for bad in ("data_msgr", "../x", "data_msgr/private", "work"):
            self.assertEqual(self.get("/api/data/report?folder=%s&size=s10m" % bad)[0], 400, bad)

    def test_link_into_private_is_refused(self):
        if not self.links:
            self.skipTest("no symlinks here")
        st, d, raw = self.get("/api/data/report?folder=data_chat&size=s10m")      # data_chat is a link into data_msgr/private in the test layout
        self.assertEqual(st, 403)
        self.assertNotIn(SECRET.encode(), raw)


class Assets(DataBase):
    def test_every_script_the_pages_import_is_served(self):
        """U3b shipped personas.js without a STATIC entry, so the browser got a 404 and no page rendered. Every .js file and every ./x.js import must be served."""
        import re
        folder = UI.STATIC_DIR
        for name in sorted(os.listdir(folder)):
            if name.endswith((".js", ".css", ".svg")):
                self.assertIn("/static/" + name, UI.STATIC, name)
                self.assertEqual(self.api("GET", "/static/" + name, token=False)[0], 200, name)
        for name in sorted(os.listdir(folder)):
            if name.endswith(".js"):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    for m in re.finditer(r"from\s+'\./([\w.-]+)'", f.read()):
                        self.assertIn("/static/" + m.group(1), UI.STATIC, "%s imports %s" % (name, m.group(1)))


class Hub(unittest.TestCase):
    def test_sizes_match_the_model_presets(self):
        from persona_lm.model import PRESETS
        self.assertIn(UD.DEFAULT_SIZE, PRESETS)


if __name__ == "__main__":
    unittest.main()
