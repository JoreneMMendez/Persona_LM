"""Stage U5a: the Training page routes (persona_lm/ui_train.py). Read only: plan, checkpoints, run numbers, log."""
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from persona_lm import pipeline as P
from persona_lm import ui_server as UI
from persona_lm import ui_train as UT
from persona_lm.bpe import Tokenizer
from persona_lm.ckpt import save_ckpt
from test_ui_u0 import SECRET, Base

CORPUS = "the cat sat on the mat. luna loves mango and the dog. hello how are you today fine thanks. " * 40


def make_run(root, rel, step=800, steps=800, sha=None, val=None, log=None):
    d = os.path.join(root, *rel.split("/"))
    os.makedirs(d, exist_ok=True)
    meta = {"cfg": {"n_layers": 2, "d_model": 16, "n_heads": 2, "d_ff": 32, "ctx": 64, "vocab_size": 300}, "hp": {"steps": steps, "lr": 1e-4},
            "step": step, "history": []}
    if sha:
        meta["tokenizer_sha256"] = sha
    save_ckpt(os.path.join(d, "ckpt.npz"), {"w": np.zeros(2, dtype=np.float32)}, {}, meta)
    if val is not None:
        with open(os.path.join(d, "metrics.jsonl"), "w") as f:
            for i, v in enumerate(val, 1):
                f.write(json.dumps({"step": i * 100, "loss": 2.0 - i * 0.1, "lr": 1e-4, "val_loss": v}) + "\n")
                f.write(json.dumps({"step": i * 100 + 10, "loss": 1.9 - i * 0.1, "lr": 1e-4}) + "\n")
    if log is not None:
        with open(os.path.join(d, "train.log"), "w") as f:
            f.write(log)
    return d


class TrainBase(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        r = cls.root
        cls.tok = Tokenizer.train([CORPUS], 300)
        cls.tok.save(os.path.join(r, "work", "tok.json"))
        sha = cls.tok.fingerprint()
        make_run(r, "work3/run_good", step=800, sha=sha, val=[1.2, 1.0, 0.9, 0.91])
        make_run(r, "work3/run_bad", step=3000, steps=3000, sha="0" * 64, val=[0.9, 1.0, 1.1, 1.35], log="line one\nline two\n")
        os.makedirs(os.path.join(r, "work3", "empty_run"))
        os.makedirs(os.path.join(r, "work3", "private_x"))
        with open(os.path.join(r, "work3", "run_good", "secret.txt"), "w") as f:
            f.write(SECRET)
        with open(os.path.join(r, "work", "notes.log"), "w") as f:
            f.write(SECRET)

    def get(self, path, **kw):
        st, _, body = self.api("GET", path, **kw)
        return st, json.loads(body or b"{}")


class Routes(TrainBase):
    PATHS = ["/api/train/plan", "/api/train/checkpoints", "/api/train/history?run=work3/run_good", "/api/train/log?run=work3/run_good"]

    def test_routes_are_read_only_and_listed(self):
        for p in UT.__doc__ and UI.TRAIN_GET:
            self.assertIn(p, UI.ROUTES["GET"])
            self.assertNotIn(p, UI.ROUTES["POST"])
        self.assertEqual([p for p in UI.ROUTES["POST"] if "/train" in p], ["/api/train/start", "/api/train/stop"])     # U5b: see test_ui_u5b.py
        self.assertIn("/static/training.js", UI.ROUTES["GET"])

    def test_need_token_and_host(self):
        for p in self.PATHS:
            self.assertEqual(self.api("GET", p, token=False)[0], 401, p)
            self.assertEqual(self.api("GET", p, token="wrong")[0], 401, p)
            self.assertEqual(self.api("GET", p, host="evil.example.com")[0], 403, p)
            self.assertEqual(self.api("GET", p, headers={"Origin": "http://evil.example.com"})[0], 403, p)
            self.assertEqual(self.api("GET", p)[0], 200, p)

    def test_post_is_refused(self):
        for p in ("/api/train/plan", "/api/train/checkpoints", "/api/train/history", "/api/train/log"):          # the read-only routes take no POST
            self.assertEqual(self.api("POST", p, {"cmd": "train"})[0], 404, p)

    def test_stage_and_section(self):
        st, info = self.get("/api/info")
        self.assertEqual(info["stage"], "U8")
        self.assertTrue([s for s in info["sections"] if s["id"] == "training"][0]["ready"])

    def test_module_never_writes_or_starts_programs(self):
        src = open(UT.__file__, encoding="utf-8").read()
        self.assertNotRegex(src, r"^\s*(import|from)\s+(subprocess|ctypes|pty|threading|socket)\b")
        self.assertNotRegex(src, r"os\.(system|popen|remove|rename|replace|mkdir|makedirs|exec\w*|spawn\w*)\(")
        self.assertNotRegex(src, r"open\([^)]*[\"'][wax]b?\+?[\"']")


class Plan(TrainBase):
    def test_no_project_gives_init_command(self):
        st, j = self.get("/api/train/plan?workdir=work2")
        self.assertEqual(st, 200)
        self.assertFalse(j["exists"])
        self.assertIn("pipeline init", j["init_command"])

    def test_steps_and_next(self):
        proj = {**P.DEFAULTS, "chat_train": "data_msgr/train.jsonl", "chat_val": "data_msgr/val.jsonl", "size": "s30m", "languages": ["ceb", "en"]}
        P.save_project(os.path.join(self.root, "work"), proj)
        os.remove(os.path.join(self.root, "work", "tok.json")) if False else None
        st, j = self.get("/api/train/plan?workdir=work")
        self.assertEqual(st, 200)
        self.assertTrue(j["exists"])
        by = {s["name"]: s for s in j["steps"]}
        self.assertEqual(by["tokenizer"]["state"], "done")                  # tok.json exists
        self.assertEqual(by["pretrain"]["state"], "skipped")
        self.assertEqual(j["next"], "chat-data")
        self.assertEqual(by["chat-data"]["state"], "next")
        self.assertIn("python -m persona_lm prepare", by["chat-data"]["command"])
        self.assertIsNone(by["finetune"]["command"])                        # not until the chat tokens exist
        self.assertIn("chat", j["chat_command"])
        self.assertEqual(j["project"]["size"], "s30m")
        self.assertFalse(j["project"]["decisions_confirmed"])
        self.assertNotIn(SECRET, json.dumps(j))

    def test_bad_workdir_refused(self):
        for w in ("../x", "data_msgr", "work/../data_msgr", "C:\\Windows", "/etc"):
            st, j = self.get("/api/train/plan?workdir=" + w.replace("\\", "%5C"))
            self.assertEqual(st, 400, w)

    def test_damaged_project_is_a_message_not_a_crash(self):
        os.makedirs(os.path.join(self.root, "work2"), exist_ok=True)
        with open(os.path.join(self.root, "work2", "project.json"), "w") as f:
            f.write("{not json")
        st, j = self.get("/api/train/plan?workdir=work2")
        self.assertEqual(st, 200)
        self.assertIn("could not be read", j["error"])
        os.remove(os.path.join(self.root, "work2", "project.json"))


class Checkpoints(TrainBase):
    def test_lists_only_runs_with_a_checkpoint(self):
        st, j = self.get("/api/train/checkpoints")
        self.assertEqual(st, 200)
        names = [r["run"] for r in j["runs"]]
        self.assertEqual(names, sorted(names))
        self.assertIn("work3/run_bad", names)
        self.assertIn("work3/run_good", names)
        self.assertNotIn("work3/empty_run", names)
        self.assertNotIn("work3/private_x", names)

    def test_numbers(self):
        _, j = self.get("/api/train/checkpoints")
        by = {r["run"]: r for r in j["runs"]}
        good, bad = by["work3/run_good"], by["work3/run_bad"]
        self.assertEqual((good["step"], good["steps_planned"]), (800, 800))
        self.assertEqual(good["model"]["n_layers"], 2)
        self.assertEqual(good["val"]["best_step"], 300)
        self.assertFalse(good["val"]["rising"])                              # 0.91 vs best 0.90 is within 3 percent
        self.assertTrue(bad["val"]["rising"])
        self.assertEqual(bad["val"]["best"], 0.9)
        self.assertIs(good["tokenizer_ok"], True)
        self.assertIs(bad["tokenizer_ok"], False)
        self.assertTrue(j["tokenizer_present"])

    def test_no_secret_and_no_absolute_paths(self):
        st, _, body = self.api("GET", "/api/train/checkpoints")
        self.assertNotIn(SECRET.encode(), body)
        self.assertNotIn(self.root.encode(), body)

    def test_damaged_checkpoint_is_listed_as_unreadable(self):
        d = os.path.join(self.root, "work3", "run_dead")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "ckpt.npz"), "wb") as f:
            f.write(b"not a zip")
        _, j = self.get("/api/train/checkpoints")
        dead = [r for r in j["runs"] if r["run"] == "work3/run_dead"][0]
        self.assertFalse(dead["readable"])
        self.assertNotIn("step", dead)


class HistoryAndLog(TrainBase):
    def test_history(self):
        st, j = self.get("/api/train/history?run=work3/run_good")
        self.assertEqual(st, 200)
        self.assertEqual(j["rows"], 8)
        self.assertEqual(len(j["val"]), 4)
        self.assertEqual(j["best"]["step"], 300)
        self.assertEqual([v["step"] for v in j["val"] if v["best"]], [300])
        self.assertEqual(j["last_step"], 410)

    def test_history_is_thinned(self):
        d = make_run(self.root, "work3/run_long", val=None)
        with open(os.path.join(d, "metrics.jsonl"), "w") as f:
            for i in range(1, 3001):
                f.write(json.dumps({"step": i, "loss": 1.0}) + "\n")
        _, j = self.get("/api/train/history?run=work3/run_long")
        self.assertLessEqual(len(j["train"]), UT.MAX_POINTS)
        self.assertEqual(j["train"][-1]["step"], 3000)
        self.assertEqual(j["rows"], 3000)

    def test_run_name_must_be_a_run_in_a_work_folder(self):
        for run in ("", "work3", "work3/", "work3/../../x", "data_msgr/private", "data_msgr/x", "../work3/run_good", "work3/run_good/x", "identity_core/a",
                    "C:%5Cx", "%2Fetc%2Fpasswd", "work3/private_x/../run_good"):
            for route in ("history", "log"):
                st, j = self.get("/api/train/%s?run=%s" % (route, run))
                self.assertIn(st, (400, 403), (route, run))
        st, _ = self.get("/api/train/history?run=work3/nope")
        self.assertEqual(st, 404)

    def test_log_from_metrics(self):
        _, j = self.get("/api/train/log?run=work3/run_good")
        self.assertEqual(j["source"], "metrics")
        self.assertEqual(len(j["lines"]), 8)
        self.assertIn("val 1.2000", j["lines"][0])
        self.assertEqual(j["files"], [])

    def test_log_file_tail(self):
        _, j = self.get("/api/train/log?run=work3/run_bad&file=train.log")
        self.assertEqual((j["source"], j["lines"]), ("file", ["line one", "line two"]))
        self.assertIn("train.log", j["files"])

    def test_log_file_long_is_cut(self):
        d = os.path.join(self.root, "work3", "run_bad")
        with open(os.path.join(d, "big.log"), "w") as f:
            for i in range(20000):
                f.write("step %d loss 1.0\n" % i)
        _, j = self.get("/api/train/log?run=work3/run_bad&file=big.log")
        self.assertLessEqual(len(j["lines"]), UT.LOG_TAIL_LINES)
        self.assertEqual(j["lines"][-1], "step 19999 loss 1.0")
        self.assertTrue(j["cut"])
        self.assertTrue(re.match(r"^step \d+ loss 1\.0$", j["lines"][0]))     # a half line was dropped

    def test_only_log_files_in_the_run_folder(self):
        for f in ("secret.txt", "../notes.log", "..%2Fnotes.log", "..%5Cnotes.log", "ckpt.npz", "metrics.jsonl", "nope.log", "%2Fetc%2Fhosts"):
            st, _, body = self.api("GET", "/api/train/log?run=work3/run_good&file=" + f)
            self.assertEqual(st, 404, f)
            self.assertNotIn(SECRET.encode(), body)

    def test_model_or_chat_text_cannot_reach_any_route(self):
        st, _, body = self.api("GET", "/api/train/plan?workdir=work&x=/approve%201")
        self.assertEqual(st, 200)
        self.assertEqual(self.api("POST", "/api/train/log", {"message": "/approve 1"})[0], 404)


class Static(unittest.TestCase):
    def test_page_files_and_wiring(self):
        d = UI.STATIC_DIR
        self.assertTrue(os.path.exists(os.path.join(d, "training.js")))
        app = open(os.path.join(d, "app.js"), encoding="utf-8").read()
        self.assertIn("from './training.js'", app)
        self.assertIn("training: (main) => trainingPage(main)", app)
        js = open(os.path.join(d, "training.js"), encoding="utf-8").read()
        js_no_ns = js.replace("http://www.w3.org/2000/svg", "")                   # U5b: the SVG namespace name (an identifier, never fetched) for the loss chart
        self.assertNotRegex(js_no_ns, r"innerHTML|insertAdjacentHTML|eval\(|https?://")
        self.assertNotIn(": null);", js)                                       # a statement ending like this was an append(null), which prints the word null
        self.assertNotRegex(js, r"method:\s*['\"]POST")                     # the page never sets its own method: posts go through api(path, body)


if __name__ == "__main__":
    unittest.main()
