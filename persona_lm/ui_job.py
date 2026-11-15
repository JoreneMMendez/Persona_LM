"""Stage U5b: start and stop a training step from the web UI, as a background process.

  GET  /api/train/job?workdir=work   the current job (state, elapsed, newest log lines) + which pipeline steps can be started now
  POST /api/train/start              {"workdir": "work", "step": "finetune", "confirm": true}
  POST /api/train/stop               {} = ask the trainer to write a checkpoint and stop; {"force": true} = end the process now

What this can and cannot do (tests in tests/test_ui_u5b.py):
  * The command is NEVER taken from the page or from model text. Only a step name is sent; the arguments come from the project's own
    pipeline plan (pipeline.build_steps), the same ones the terminal shows. Only the NEXT step can start. The tokenizer step
    (it freezes work/tok.json) is not startable here: terminal only.
  * Every file the step reads or writes must lie inside the project folder and pass PathGuard (no data_msgr/private, nothing outside).
    A step that names a file outside the project (typical for the public pretraining text) is shown with its command and stays a terminal job.
  * One job at a time over all work folders. The decisions must be confirmed (`pipeline confirm`, terminal) and work/tok.json must match
    the project's fingerprint, the same checks as `pipeline run`.
  * The process is persona_lm/ui_jobrun.py, which runs the normal CLI. Stop = a stop file: the trainer saves a checkpoint and ends (same as
    Ctrl+C). Force stop = terminate (progress since the last checkpoint is lost).
  * Files written: <workdir>/ui_job.json (state + heartbeat), the log (<run>/ui_train.log for training, <workdir>/ui_prepare.log for prepare),
    <workdir>/ui_job.stop while a stop is asked.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time

from . import pipeline as P
from . import report as R
from .bpe import Tokenizer
from .ui_train import TrainProblem, WORKDIRS

STEPS = ("pretrain-data", "pretrain", "chat-data", "finetune")
TRAIN_STEPS = ("pretrain", "finetune")
STALE_SECONDS = 120
LOG_TAIL_BYTES = 16384
LOG_TAIL_LINES = 40
PATH_FLAGS = ("--data", "--val-data", "--tokenizer", "--out", "--run", "--data-dir", "--init-from")
LABELS = {"pretrain-data": "Prepare the public text", "pretrain": "Pretrain", "chat-data": "Prepare your chats", "finetune": "Fine-tune on your chats"}


class JobProblem(TrainProblem):
    pass


def _paths_in(argv):
    """The file arguments of a CLI argv: every value after a path flag, up to the next --flag."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] in PATH_FLAGS:
            j = i + 1
            while j < len(argv) and not str(argv[j]).startswith("--"):
                out.append(argv[j])
                j += 1
            i = j
        else:
            i += 1
    return out


def _flag(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


class JobHub:
    def __init__(self, guard, python=None, wrapper=None):
        self.guard = guard
        self.python = python or sys.executable
        # tests replace this list with a fake program that takes the same arguments (JOBFILE STOPFILE -- ARGV...)
        self.wrapper = wrapper or [self.python, "-u", "-m", "persona_lm.ui_jobrun"]
        self.proc = None

    # ---------------------------------------------------------------- small helpers
    def _clean(self, e):
        return str(e).replace(self.guard.root, ".")

    def _show(self, argv):
        """The command as the terminal needs it, run from the project folder: the project path is left out."""
        return P.show(argv).replace(self.guard.root + os.sep, "")

    def _path(self, rel):
        try:
            return self.guard.resolve(rel)
        except Exception as e:
            raise JobProblem(self._clean(e), 403)

    def _jobfile(self, wd):
        return self._path(wd + "/ui_job.json")

    def _stopfile(self, wd):
        return self._path(wd + "/ui_job.stop")

    @staticmethod
    def _load(path):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            return rec if isinstance(rec, dict) else None
        except (OSError, ValueError):
            return None

    def _inside(self, p):
        """True when the file argument lies in the project and passes PathGuard."""
        try:
            full = os.path.realpath(p if os.path.isabs(p) else os.path.join(self.guard.root, p))
            rel = os.path.relpath(full, self.guard.root)
            if rel.startswith("..") or os.path.isabs(rel) or rel == ".":
                return False
            self.guard.resolve(rel)
            return True
        except Exception:
            return False

    def _tail(self, rel):
        if not rel:
            return []
        try:
            path = self._path(rel)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                if size > LOG_TAIL_BYTES:
                    f.seek(size - LOG_TAIL_BYTES)
                raw = f.read(LOG_TAIL_BYTES)
        except (OSError, JobProblem):
            return []
        lines = raw.decode("utf-8", "replace").replace("\r", "\n").split("\n")
        if size > LOG_TAIL_BYTES:
            lines = lines[1:]
        return [self._clean(x) for x in lines if x.strip()][-LOG_TAIL_LINES:]

    # ---------------------------------------------------------------- the job file
    def _effective(self, wd, rec):
        st = rec.get("state")
        if st in ("starting", "running"):
            if time.time() - float(rec.get("beat") or 0) > STALE_SECONDS:
                return "lost"
            return "stopping" if os.path.exists(self._stopfile(wd)) else st
        return st or "unknown"

    def _jobs(self):
        out = []
        for wd in WORKDIRS:
            try:
                rec = self._load(self._jobfile(wd)) if os.path.isdir(self._path(wd)) else None
            except JobProblem:
                rec = None
            if rec:
                rec["workdir"] = wd
                rec["state"] = self._effective(wd, rec)
                out.append(rec)
        return out

    def _current(self):
        jobs = self._jobs()
        live = [j for j in jobs if j["state"] in ("starting", "running", "stopping")]
        if live:
            return live[0]
        return max(jobs, key=lambda j: j.get("started") or 0) if jobs else None

    def _public(self, rec):
        now = time.time()
        end = rec.get("ended") or (now if rec["state"] in ("starting", "running", "stopping") else rec.get("beat") or now)
        out = {k: rec.get(k) for k in ("workdir", "step", "title", "run", "state", "exit_code", "command", "planned_steps")}
        out["started"] = int(rec.get("started") or 0)
        out["elapsed"] = max(0, int(end - (rec.get("started") or end)))
        out["log"] = self._tail(rec.get("log"))
        out["log_file"] = rec.get("log")
        return out

    # ---------------------------------------------------------------- what can start
    def _steps_info(self, wd):
        full = self._path(wd)
        out = {"workdir": wd, "exists": False, "steps": []}
        if not os.path.exists(P.project_path(full)):
            out["why"] = "No project plan in %s yet (see the Pipeline plan above)." % wd
            return out
        try:
            proj = P.load_project(full)
            steps = P.build_steps(proj, full)
        except Exception as e:
            out["why"] = self._clean(e)
            return out
        out["exists"] = True
        nxt = P.next_step(steps)
        out["confirmed"] = bool(proj["decisions_confirmed"])
        for s in steps:
            if s["name"] not in STEPS:
                continue
            info = {"name": s["name"], "title": LABELS[s["name"]], "state": "done" if s["done"] else "skipped" if s["skipped"] else "next" if s is nxt else "todo",
                    "command": self._show(s["argv"]) if s.get("argv") else None, "can_start": False, "reason": "", "resume_step": None, "planned_steps": None}
            if s["name"] in TRAIN_STEPS and s.get("argv"):
                info["planned_steps"] = int(_flag(s["argv"], "--steps") or 0) or None
                run = _flag(s["argv"], "--run")
                meta = R.read_ckpt_meta(os.path.join(run, "ckpt.npz")) if run else None
                if meta and isinstance(meta.get("step"), int) and meta["step"] > 0:
                    info["resume_step"] = meta["step"]
            info["can_start"], info["reason"] = self._check(proj, wd, full, s, nxt)
            out["steps"].append(info)
        return out

    def _check(self, proj, wd, full, s, nxt):
        if s["done"]:
            return False, "already done"
        if s["skipped"]:
            return False, s.get("note") or "skipped"
        if s.get("argv") is None:
            return False, s.get("note") or "not ready yet"
        if s is not nxt:
            return False, "not the next step: finish the one marked NEXT first"
        if not proj["decisions_confirmed"]:
            return False, "confirm the decisions first (terminal: `pipeline confirm`)"
        tok = os.path.join(full, "tok.json")
        if not os.path.exists(tok):
            return False, "there is no tokenizer yet (terminal only: it freezes the vocabulary)"
        try:
            if proj["tokenizer_sha256"] and Tokenizer.load(tok).fingerprint() != proj["tokenizer_sha256"]:
                return False, "tok.json is not the tokenizer this project froze: restore the original file"
        except Exception:
            return False, "tok.json cannot be read"
        outside = [p for p in _paths_in(s["argv"]) if not self._inside(p)]
        if outside:
            return False, "this step uses a file outside the project or in a private folder; run its command in the terminal"
        return True, ""

    def status(self, workdir="work"):
        if workdir not in WORKDIRS:
            raise JobProblem("unknown work folder (choose one of: %s)" % ", ".join(WORKDIRS))
        cur = self._current()
        busy = bool(cur and cur["state"] in ("starting", "running", "stopping"))
        out = {"job": self._public(cur) if cur else None, "busy": busy, "plan": self._steps_info(workdir)}
        out["note"] = ("Stop asks the trainer to save a checkpoint and end, like Ctrl+C. Run the same step again to resume from that checkpoint. "
                       "The trainer keeps only the newest two checkpoints.")
        return out

    # ---------------------------------------------------------------- start
    def start(self, body):
        wd, name = body.get("workdir"), body.get("step")
        if wd not in WORKDIRS:
            raise JobProblem("unknown work folder (choose one of: %s)" % ", ".join(WORKDIRS))
        if name == "tokenizer":
            raise JobProblem("The tokenizer step freezes work/tok.json for every model. It is not started from the web page; use the command in the terminal.", 403)
        if name not in STEPS:
            raise JobProblem("unknown step (choose one of: %s)" % ", ".join(STEPS))
        if body.get("confirm") is not True:
            raise JobProblem("confirmation missing: send confirm = true (the page asks you first)", 400)
        cur = self._current()
        if cur and cur["state"] in ("starting", "running", "stopping"):
            raise JobProblem("a job is already running (%s in %s). Stop it first." % (cur.get("step"), cur["workdir"]), 409)
        full = self._path(wd)
        if not os.path.exists(P.project_path(full)):
            raise JobProblem("no project plan in %s" % wd, 404)
        try:
            proj = P.load_project(full)
            steps = P.build_steps(proj, full)
        except Exception as e:
            raise JobProblem(self._clean(e), 422)
        nxt = P.next_step(steps)
        step = next((s for s in steps if s["name"] == name), None)
        ok, why = self._check(proj, wd, full, step, nxt) if step else (False, "unknown step")
        if not ok:
            raise JobProblem("This step cannot start: " + why, 409)
        argv = [str(a) for a in step["argv"]]
        if not proj["tokenizer_sha256"]:                                          # same as pipeline run: remember the tokenizer from now on
            proj["tokenizer_sha256"] = Tokenizer.load(os.path.join(full, "tok.json")).fingerprint()
            P.save_project(full, proj)
        run = _flag(argv, "--run")
        if name in TRAIN_STEPS:
            run_rel = wd + "/" + os.path.basename(run.rstrip("/\\"))
            run_full = self._path(run_rel)
            os.makedirs(run_full, exist_ok=True)
            log_rel = run_rel + "/ui_train.log"
        else:
            run_rel, log_rel = None, wd + "/ui_prepare.log"
        jobfile, stopfile, logfile = self._jobfile(wd), self._stopfile(wd), self._path(log_rel)
        try:
            os.remove(stopfile)
        except OSError:
            pass
        now = time.time()
        rec = {"state": "starting", "step": name, "title": LABELS[name], "run": run_rel, "log": log_rel, "command": self._show(step["argv"]),
               "planned_steps": int(_flag(argv, "--steps") or 0) or None, "started": now, "beat": now}
        with open(jobfile, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        env = dict(os.environ)
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kw["start_new_session"] = True
        with open(logfile, "wb") as lf:
            lf.write(("[web UI] %s\n" % rec["command"]).encode("utf-8", "replace"))
        lf = open(logfile, "ab")
        try:
            self.proc = subprocess.Popen(self.wrapper + [jobfile, stopfile, "--"] + argv, cwd=self.guard.root, env=env, stdin=subprocess.DEVNULL,
                                         stdout=lf, stderr=subprocess.STDOUT, **kw)
        except OSError as e:
            rec.update(state="failed", exit_code=-1, ended=time.time())
            with open(jobfile, "w", encoding="utf-8") as f:
                json.dump(rec, f)
            raise JobProblem("could not start the process: " + self._clean(e), 500)
        finally:
            lf.close()
        return {"started": True, "job": self._public({**rec, "workdir": wd, "state": "starting"})}

    # ---------------------------------------------------------------- stop
    def stop(self, body):
        cur = self._current()
        if not cur or cur["state"] not in ("starting", "running", "stopping"):
            raise JobProblem("no job is running", 409)
        wd = cur["workdir"]
        if body.get("force") is True:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
            elif cur.get("pid"):
                try:
                    os.kill(int(cur["pid"]), signal.SIGTERM)
                except (OSError, ValueError):
                    pass
            rec = self._load(self._jobfile(wd)) or {}
            rec.update(state="stopped", ended=time.time(), exit_code=-15, forced=True)
            with open(self._jobfile(wd), "w", encoding="utf-8") as f:
                json.dump(rec, f)
            try:
                os.remove(self._stopfile(wd))
            except OSError:
                pass
            return {"stopped": "forced", "job": self._public({**rec, "workdir": wd})}
        with open(self._stopfile(wd), "w") as f:
            f.write("stop\n")
        return {"stopped": "asked", "job": self._public({**cur, "state": "stopping"})}
