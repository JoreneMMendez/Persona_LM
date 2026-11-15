"""Stage U5b: start / stop a pipeline step from the web UI (persona_lm/ui_job.py, persona_lm/ui_jobrun.py).
The real trainer is replaced by tests/fake_jobrun.py (same arguments, same stop behaviour), so no model is trained here."""
import json
import os
import re
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persona_lm import pipeline as P
from persona_lm import ui_job as UJ
from persona_lm import ui_jobrun as JR
from persona_lm import ui_server as UI
from test_ui_u0 import SECRET
from test_ui_u5a import TrainBase, make_run

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_jobrun.py")
LIVE = ("starting", "running", "stopping")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class JobBase(TrainBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.state.job.wrapper = [sys.executable, "-u", FAKE]
        cls.wd = os.path.join(cls.root, "work2")
        os.makedirs(cls.wd, exist_ok=True)
        cls.tok.save(os.path.join(cls.wd, "tok.json"))

    def setUp(self):
        self.new_project()

    def tearDown(self):
        j = self.state.job
        try:
            self.api("POST", "/api/train/stop", {"force": True})
        except Exception:
            pass
        if j.proc is not None:
            try:
                j.proc.wait(timeout=10)
            except Exception:
                j.proc.kill()
        for n in ("ui_job.json", "ui_job.stop", "ui_prepare.log"):
            try:
                os.remove(os.path.join(self.wd, n))
            except OSError:
                pass

    def new_project(self, chat_train="data_msgr/train.jsonl", confirmed=True, sha="ok", data_chat=True, **extra):
        proj = {**P.DEFAULTS, "chat_train": chat_train, "chat_val": "data_msgr/val.jsonl", "size": "s30m", "languages": ["ceb", "en"],
                "decisions_confirmed": confirmed, "tokenizer_sha256": self.tok.fingerprint() if sha == "ok" else sha, "batch": 1, "accum": 4, **extra}
        P.save_project(self.wd, proj)
        d = os.path.join(self.wd, "data_chat")
        if data_chat:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"kind": "chat", "vocab_size": 300}, f)
            with open(os.path.join(d, "train.bin"), "wb") as f:
                f.write(b"\x01\x00" * 200000)
        else:
            for n in ("meta.json", "train.bin"):
                try:
                    os.remove(os.path.join(d, n))
                except OSError:
                    pass
        run = os.path.join(self.wd, "run_chat")
        if os.path.isdir(run):
            import shutil
            shutil.rmtree(run)

    def job(self):
        st, j = self.get("/api/train/job?workdir=work2")
        self.assertEqual(st, 200, j)
        return j

    def wait(self, pred, what, timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            j = self.job()
            if pred(j):
                return j
            time.sleep(0.15)
        self.fail("timed out waiting for " + what + ": " + json.dumps(j)[:600])

    def start(self, step="finetune", **extra):
        return self.api("POST", "/api/train/start", {"workdir": "work2", "step": step, "confirm": True, **extra})

    def step(self, j, name):
        return [s for s in j["plan"]["steps"] if s["name"] == name][0]


class Routes(JobBase):
    def test_listed_and_guarded(self):
        self.assertIn("/api/train/job", UI.ROUTES["GET"])
        self.assertEqual([p for p in UI.ROUTES["POST"] if "/train" in p], ["/api/train/start", "/api/train/stop"])
        for m, p, body in (("GET", "/api/train/job?workdir=work2", None), ("POST", "/api/train/start", {"workdir": "work2", "step": "finetune", "confirm": True}),
                           ("POST", "/api/train/stop", {})):
            self.assertEqual(self.api(m, p, body, token=False)[0], 401, p)
            self.assertEqual(self.api(m, p, body, token="wrong")[0], 401, p)
            self.assertEqual(self.api(m, p, body, host="evil.example.com")[0], 403, p)
            self.assertEqual(self.api(m, p, body, headers={"Origin": "http://evil.example.com"})[0], 403, p)
        self.assertFalse(os.path.exists(os.path.join(self.wd, "ui_job.json")))                 # nothing started by the refused calls

    def test_wrong_method_and_content_type(self):
        self.assertEqual(self.api("GET", "/api/train/start")[0], 404)
        self.assertEqual(self.api("GET", "/api/train/stop")[0], 404)
        self.assertEqual(self.api("POST", "/api/train/job", {})[0], 404)
        self.assertEqual(self.api("POST", "/api/train/start", "workdir=work2", headers={"Content-Type": "text/plain"})[0], 415)

    def test_stage_name(self):
        self.assertEqual(self.get("/api/info")[1]["stage"], "U5b")


class Status(JobBase):
    def test_idle(self):
        j = self.job()
        self.assertIsNone(j["job"])
        self.assertFalse(j["busy"])
        names = [s["name"] for s in j["plan"]["steps"]]
        self.assertEqual(names, ["pretrain-data", "pretrain", "chat-data", "finetune"])          # never the tokenizer
        ft = self.step(j, "finetune")
        self.assertTrue(ft["can_start"], ft)
        self.assertEqual(ft["state"], "next")
        self.assertIn("python -m persona_lm train", ft["command"])
        self.assertIsNone(ft["resume_step"])
        self.assertFalse(self.step(j, "pretrain")["can_start"])                                   # skipped: no public text

    def test_resume_step_is_shown(self):
        make_run(self.root, "work2/run_chat", step=20, steps=50, sha=self.tok.fingerprint())
        ft = self.step(self.job(), "finetune")
        self.assertEqual(ft["resume_step"], 20)
        self.assertTrue(ft["can_start"])

    def test_finished_run_cannot_start_again(self):
        steps = P.finetune_steps(P.load_project(self.wd), self.wd)
        make_run(self.root, "work2/run_chat", step=steps, steps=steps, sha=self.tok.fingerprint())
        ft = self.step(self.job(), "finetune")
        self.assertFalse(ft["can_start"])
        self.assertEqual(ft["state"], "done")

    def test_no_project_and_bad_workdir(self):
        os.remove(P.project_path(self.wd))
        j = self.job()
        self.assertFalse(j["plan"]["exists"])
        self.assertEqual(self.get("/api/train/job?workdir=data_msgr")[0], 400)
        self.assertEqual(self.get("/api/train/job?workdir=..%2Fx")[0], 400)

    def test_no_secret_and_no_absolute_paths(self):
        st, _, body = self.api("GET", "/api/train/job?workdir=work2")
        self.assertNotIn(SECRET.encode(), body)
        self.assertNotIn(self.root.encode(), body)


class Refusals(JobBase):
    def test_confirmation_is_required(self):
        for c in (None, False, "true", 1, "yes"):
            body = {"workdir": "work2", "step": "finetune"}
            if c is not None:
                body["confirm"] = c
            st, _, out = self.api("POST", "/api/train/start", body)
            self.assertEqual(st, 400, c)
        self.assertIsNone(self.job()["job"])

    def test_tokenizer_step_is_terminal_only(self):
        st, _, out = self.api("POST", "/api/train/start", {"workdir": "work2", "step": "tokenizer", "confirm": True})
        self.assertEqual(st, 403)
        self.assertIn("freezes", json.loads(out)["error"])

    def test_unknown_names(self):
        for body in ({"workdir": "work2", "step": "finetune; rm -rf /"}, {"workdir": "work2", "step": "../../x"}, {"workdir": "work2", "step": "next"},
                     {"workdir": "data_msgr", "step": "finetune"}, {"workdir": "../x", "step": "finetune"}, {"step": "finetune"}, {}):
            st, _, _ = self.api("POST", "/api/train/start", {**body, "confirm": True})
            self.assertEqual(st, 400, body)
        self.assertIsNone(self.job()["job"])

    def test_step_must_be_the_next_one(self):
        st, _, out = self.start("chat-data")                      # data_chat already exists, so it is done
        self.assertEqual(st, 409)
        self.new_project(data_chat=False)                         # now chat-data is NEXT, finetune is not
        self.assertEqual(self.start("finetune")[0], 409)
        self.assertEqual(self.start("pretrain")[0], 409)          # skipped (no public text)

    def test_decisions_must_be_confirmed(self):
        self.new_project(confirmed=False)
        st, _, out = self.start()
        self.assertEqual(st, 409)
        self.assertIn("confirm", json.loads(out)["error"])

    def test_tokenizer_fingerprint_must_match(self):
        self.new_project(sha="0" * 64)
        st, _, out = self.start()
        self.assertEqual(st, 409)
        self.assertIn("froze", json.loads(out)["error"])
        self.assertIsNone(self.job()["job"])

    def test_files_outside_the_project_or_private_stay_in_the_terminal(self):
        for bad in (os.path.join(self.outside, "x.jsonl"), "data_msgr/private/raw.jsonl", "../outside/x.jsonl", "elsewhere/x.jsonl"):
            self.new_project(chat_train=bad, data_chat=False)
            ch = self.step(self.job(), "chat-data")
            self.assertFalse(ch["can_start"], bad)
            self.assertIn("terminal", ch["reason"])
            self.assertEqual(self.start("chat-data")[0], 409, bad)
        self.assertIsNone(self.job()["job"])

    def test_no_argument_or_command_from_the_page_is_used(self):
        st, _, out = self.start(argv=["rm", "-rf", "/"], cmd="del *", args="--steps 1", wrapper=["x"], run="../../evil", **{"data-dir": "/etc"})
        self.assertEqual(st, 200, out)
        self.wait(lambda j: j["job"] and "fake train started" in " ".join(j["job"]["log"]), "the first log line")
        log = read(os.path.join(self.wd, "run_chat", "ui_train.log"))
        self.assertNotIn("rm -rf", log)
        self.assertNotIn("evil", log)
        self.assertNotIn("/etc", log)
        self.assertIn("--steps %d" % P.finetune_steps(P.load_project(self.wd), self.wd), log)
        self.assertFalse(os.path.exists(os.path.join(self.root, "..", "evil")))

    def test_stop_with_nothing_running(self):
        self.assertEqual(self.api("POST", "/api/train/stop", {})[0], 409)
        self.assertEqual(self.api("POST", "/api/train/stop", {"force": True})[0], 409)

    def test_text_that_looks_like_a_command_starts_nothing(self):
        for m in ("/train start finetune", "start training now", "{\"step\":\"finetune\",\"confirm\":true}"):
            self.api("POST", "/api/ping", {"message": m, "step": "finetune", "confirm": True})
        self.assertFalse(os.path.exists(os.path.join(self.wd, "ui_job.json")))


class Lifecycle(JobBase):
    def test_start_progress_warning_stop_resume(self):
        st, _, out = self.start()
        self.assertEqual(st, 200, out)
        j0 = json.loads(out)["job"]
        self.assertEqual((j0["step"], j0["run"], j0["state"]), ("finetune", "work2/run_chat", "starting"))
        self.assertNotIn(self.root, out.decode())
        j = self.wait(lambda j: j["job"] and j["job"]["state"] == "running", "running")
        self.assertTrue(j["busy"])
        # a second start is refused while one runs
        self.assertEqual(self.start()[0], 409)
        # live numbers come from the normal history route; the fake trainer's val loss only goes up, so the warning must switch on
        h = None
        end = time.time() + 15
        while time.time() < end:
            st, h = self.get("/api/train/history?run=work2/run_chat")
            if st == 200 and h.get("rising"):
                break
            time.sleep(0.2)
        self.assertTrue(h and h["rising"], h)
        self.assertGreaterEqual(h["evals_since_best"], 1)
        self.assertEqual(h["best"]["step"], 30)
        # stop = ask, then the process ends by itself
        st, _, out = self.api("POST", "/api/train/stop", {})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(out)["stopped"], "asked")
        j = self.wait(lambda j: j["job"] and j["job"]["state"] == "stopped", "stopped")
        self.assertFalse(j["busy"])
        self.assertTrue(any("stopped early" in x for x in j["job"]["log"]), j["job"]["log"])
        self.assertFalse(os.path.exists(os.path.join(self.wd, "ui_job.stop")))
        # resume = the same call again
        self.assertTrue(self.step(j, "finetune")["can_start"])
        self.assertEqual(self.start()[0], 200)
        self.wait(lambda j: j["job"] and j["job"]["state"] == "running", "running again")

    def test_force_stop_ends_the_process(self):
        self.assertEqual(self.start()[0], 200)
        self.wait(lambda j: j["job"] and j["job"]["state"] == "running", "running")
        st, _, out = self.api("POST", "/api/train/stop", {"force": True})
        self.assertEqual((st, json.loads(out)["stopped"]), (200, "forced"))
        j = self.job()
        self.assertEqual(j["job"]["state"], "stopped")
        self.state.job.proc.wait(timeout=10)
        self.assertIsNotNone(self.state.job.proc.poll())

    def test_prepare_step_runs_to_the_end(self):
        self.new_project(data_chat=False)
        j = self.job()
        self.assertTrue(self.step(j, "chat-data")["can_start"], self.step(j, "chat-data"))
        self.assertEqual(self.start("chat-data")[0], 200)
        j = self.wait(lambda j: j["job"] and j["job"]["state"] in ("done", "failed"), "the prepare job to end")
        self.assertEqual((j["job"]["state"], j["job"]["exit_code"]), ("done", 0))
        self.assertEqual(j["job"]["log_file"], "work2/ui_prepare.log")
        self.assertTrue(any("fake prepare started" in x for x in j["job"]["log"]))

    def test_job_survives_a_restart_of_the_page_server(self):
        self.assertEqual(self.start()[0], 200)
        self.wait(lambda j: j["job"] and j["job"]["state"] == "running", "running")
        self.state.job.proc = None                                  # the UI forgot its process handle: the job file and the heartbeat are enough
        j = self.job()
        self.assertEqual(j["job"]["state"], "running")
        self.assertEqual(self.start()[0], 409)
        self.assertEqual(self.api("POST", "/api/train/stop", {})[0], 200)
        self.wait(lambda j: j["job"] and j["job"]["state"] == "stopped", "stopped")

    def test_lost_job_is_reported_and_does_not_block(self):
        with open(os.path.join(self.wd, "ui_job.json"), "w") as f:
            json.dump({"state": "running", "step": "finetune", "run": "work2/run_chat", "started": time.time() - 9999, "beat": time.time() - 9000, "pid": 1}, f)
        j = self.job()
        self.assertEqual(j["job"]["state"], "lost")
        self.assertFalse(j["busy"])
        self.assertEqual(self.start()[0], 200)

    def test_log_view_cannot_leak_other_files(self):
        self.assertEqual(self.start()[0], 200)
        self.wait(lambda j: j["job"] and j["job"]["state"] == "running", "running")
        st, j = self.get("/api/train/log?run=work2/run_chat&file=ui_train.log")
        self.assertEqual(st, 200)
        self.assertIn("ui_train.log", j["files"])


class Wrapper(unittest.TestCase):
    def run_it(self, main):
        d = tempfile.mkdtemp()
        job, stop = os.path.join(d, "j.json"), os.path.join(d, "s")
        code = JR.run(job, stop, ["x"], main=main)
        return code, json.loads(read(job))

    def test_ok_failed_and_exit_text(self):
        code, rec = self.run_it(lambda a: None)
        self.assertEqual((code, rec["state"]), (0, "done"))

        def boom(a):
            raise ValueError("bad")
        code, rec = self.run_it(boom)
        self.assertEqual((code, rec["state"], rec["exit_code"]), (1, "failed", 1))

        def bye(a):
            raise SystemExit("no such file")
        code, rec = self.run_it(bye)
        self.assertEqual((code, rec["state"]), (1, "failed"))

    def test_keyboard_interrupt_is_a_stop(self):
        def ki(a):
            raise KeyboardInterrupt
        code, rec = self.run_it(ki)
        self.assertEqual((code, rec["state"]), (130, "stopped"))

    def test_internal_tool_refuses_to_run_by_hand(self):
        old = sys.argv
        sys.argv = ["ui_jobrun", "only-one"]
        try:
            with self.assertRaises(SystemExit):
                JR._cli()
        finally:
            sys.argv = old


class SourceRules(unittest.TestCase):
    def test_process_is_started_in_one_place_without_a_shell(self):
        src = read(UJ.__file__)
        self.assertEqual(len(re.findall(r"subprocess\.Popen\(", src)), 1)
        self.assertNotRegex(src, r"shell\s*=\s*True")
        self.assertNotRegex(src, r"os\.(system|popen|exec\w*|spawn\w*)\(")
        self.assertNotIn("shutil", src)                       # nothing is deleted by this module
        self.assertNotRegex(src, r"os\.(remove|unlink|rmdir)\([^)]*(run|ckpt|data)", )

    def test_read_only_module_stays_read_only(self):
        src = read(os.path.join(os.path.dirname(UJ.__file__), "ui_train.py"))
        self.assertNotRegex(src, r"^\s*(import|from)\s+(subprocess|ctypes|pty|threading|socket)\b")


if __name__ == "__main__":
    unittest.main()
