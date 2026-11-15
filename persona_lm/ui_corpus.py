"""Stage U8: the "Language data" page routes of the web UI (download public text, clean it, check it).

  GET  /api/corpus/overview   per language: what is downloaded / cleaned / mixed (sizes and dates only), the steps a run would take,
                              what each project's pretraining text points at, free disk space, whether `datasets` is installed
  GET  /api/corpus/job        the current or last job: state, step N of M, newest log lines
  POST /api/corpus/start      {"languages": ["en","tl"], "mode": "all"|"clean", "mb": {"en": 420}, "mix": false, "confirm": true, "ack_taglish": true}
  POST /api/corpus/stop       {} = finish the current document and stop; {"force": true} = end the process now

What this can and cannot do (tests in tests/test_ui_corpus.py):
  * The command is NEVER taken from the page. The page sends language ids, a mode and numbers; this file validates them and builds the
    arguments of `python -m persona_lm corpus` itself. No path, URL, dataset name or program name can come from the page.
  * This is the only part of the UI whose background job reaches the internet, and only after the owner presses Start and confirms.
    It downloads PUBLIC text (FineWeb-Edu, FineWeb-2, PinoyExchange threads). Nothing of the owner's is read or uploaded.
  * It writes only inside public_text/, public_clean/ and public_mix/ (all three are public-text folders, ignored by git). Existing good
    files are replaced only when a step finished (see corpus.py).
  * One language-data job at a time. A running training job does not block it, but the page warns (both use the disk and CPU).
  * The job is persona_lm/ui_jobrun.py running the normal CLI, so it keeps going when the page or the UI is closed, and the page picks
    it up again. Files: public_text/corpus_job.json (state + heartbeat), corpus.log, corpus_progress.json, corpus_job.stop.
"""
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from . import corpus as C
from . import pipeline as P

LIVE = ("starting", "running", "stopping")
STALE_SECONDS = 120
LOG_TAIL_BYTES = 16384
LOG_TAIL_LINES = 40
JOBFILE = "public_text/corpus_job.json"
STOPFILE = "public_text/corpus_job.stop"
LOGFILE = "public_text/corpus.log"
PROGRESS = "public_text/corpus_progress.json"
WORKDIRS = ("work", "work2", "work3")
PUBLIC_DIRS = ("public_text", "public_clean", "public_mix")
DISK_FACTOR = 2.5                 # raw + cleaned + mix copies of a download
UNKNOWN_MB = 1500                 # Cebuano has no cap; a rough allowance for the free-space check
PREPARE_OUT = "data_pre_new"


class CorpusProblem(Exception):
    def __init__(self, msg, code=400):
        Exception.__init__(self, msg)
        self.code = code


class CorpusHub:
    def __init__(self, guard, train_hub=None, python=None, wrapper=None):
        self.guard = guard
        self.train = train_hub                       # the JobHub of the Training page (only asked "is training running?")
        self.python = python or sys.executable
        self.wrapper = wrapper or [self.python, "-u", "-m", "persona_lm.ui_jobrun"]     # tests replace this with a fake program
        self.proc = None

    # ---------------------------------------------------------------- small helpers
    def _clean(self, e):
        return str(e).replace(self.guard.root, ".")

    def _path(self, rel):
        try:
            return self.guard.resolve(rel)
        except Exception as e:
            raise CorpusProblem(self._clean(e), 403)

    @staticmethod
    def _load(path):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            return rec if isinstance(rec, dict) else None
        except (OSError, ValueError):
            return None

    def _file_info(self, rel):
        try:
            st = os.stat(self._path(rel))
        except (OSError, CorpusProblem):
            return {"exists": False}
        return {"exists": True, "mb": round(st.st_size / 1e6, 1), "modified": int(st.st_mtime)}

    def _tail(self, rel):
        try:
            path = self._path(rel)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                if size > LOG_TAIL_BYTES:
                    f.seek(size - LOG_TAIL_BYTES)
                raw = f.read(LOG_TAIL_BYTES)
        except (OSError, CorpusProblem):
            return []
        lines = raw.decode("utf-8", "replace").replace("\r", "\n").split("\n")
        if size > LOG_TAIL_BYTES:
            lines = lines[1:]
        return [self._clean(x) for x in lines if x.strip()][-LOG_TAIL_LINES:]

    @staticmethod
    def terminal_command(langs, mode, mb, mix):
        argv = ["python", "-m", "persona_lm", "corpus", "--lang", ",".join(langs), "--mode", mode]
        if mb:
            argv += ["--mb"] + ["%s=%d" % (k, v) for k, v in mb.items()]
        if mix:
            argv.append("--mix")
        return " ".join(argv)

    # ---------------------------------------------------------------- the job file
    def _effective(self, rec):
        st = rec.get("state")
        if st in ("starting", "running"):
            if time.time() - float(rec.get("beat") or 0) > STALE_SECONDS:
                return "lost"
            return "stopping" if os.path.exists(self._path(STOPFILE)) else st
        return st or "unknown"

    def _record(self):
        rec = self._load(self._path(JOBFILE))
        if rec:
            rec["state"] = self._effective(rec)
        return rec

    def _public(self, rec):
        now = time.time()
        live = rec["state"] in LIVE
        end = rec.get("ended") or (now if live else rec.get("beat") or now)
        prog = self._load(self._path(PROGRESS)) or {}
        out = {k: rec.get(k) for k in ("state", "langs", "mode", "mix", "command", "stages", "exit_code")}
        out["started"] = int(rec.get("started") or 0)
        out["elapsed"] = max(0, int(end - (rec.get("started") or end)))
        out["step"] = {"index": prog.get("index"), "total": prog.get("total") or len(rec.get("stages") or []), "title": prog.get("title")}
        out["log"] = self._tail(LOGFILE)
        return out

    def _training_running(self):
        try:
            cur = self.train._current() if self.train is not None else None
        except Exception:
            return False
        return bool(cur and cur.get("state") in LIVE)

    def job_status(self):
        rec = self._record()
        live = bool(rec and rec["state"] in LIVE)
        return {"job": self._public(rec) if rec else None, "busy": live, "training_running": self._training_running(),
                "note": "Stop finishes the current document and ends the run. Files from steps that did not finish are not used; your existing files stay as they were."}

    # ---------------------------------------------------------------- overview
    def _projects(self):
        out = []
        for wd in WORKDIRS:
            try:
                full = self._path(wd)
                if not os.path.exists(P.project_path(full)):
                    continue
                proj = P.load_project(full)
            except Exception:
                continue
            reads = []
            for p in proj.get("pretrain_text") or []:
                first = str(p).replace("\\", "/").strip("/").split("/")[0]
                reads.append({"path": str(p), "kind": first if first in PUBLIC_DIRS else "other"})
            out.append({"workdir": wd, "reads": reads, "has_pretrain_text": bool(reads),
                        "tokens_done": os.path.exists(os.path.join(full, "data_pre", "meta.json"))})
        return out

    def overview(self):
        langs = []
        for l in C.LANG_ORDER:
            spec = C.LANGS[l]
            raw = self._file_info(spec["raw"]) if spec["raw"] else None
            clean = self._file_info(spec["clean"])
            mix = self._file_info("public_mix/%s.jsonl" % l)
            partial = [r for r in ([spec["raw"] + ".part"] if spec["raw"] else []) + [spec["clean"] + ".part"] if self._file_info(r)["exists"]]
            state = "ready" if clean["exists"] else "downloaded" if (raw and raw["exists"]) else "missing"
            langs.append({"id": l, "label": spec["label"], "source": spec["source"], "license": spec["license"], "note": spec["note"],
                          "default_mb": spec["default_mb"], "capped": spec["capped"], "raw": raw, "clean": clean, "mix": mix,
                          "state": state, "partial": bool(partial), "needs_ack": l == "taglish"})
        try:
            free_mb = int(shutil.disk_usage(self.guard.root).free / 1e6)
        except OSError:
            free_mb = None
        rec = self._record()
        return {"languages": langs, "datasets_installed": importlib.util.find_spec("datasets") is not None, "install_command": "pip install datasets",
                "free_mb": free_mb, "projects": self._projects(), "prepare_out": PREPARE_OUT,
                "prepare_command": ("python -m persona_lm prepare --kind pretrain --tokenizer work\\tok.json --data "
                                    + " ".join("public_mix\\%s.jsonl" % l for l in C.LANG_ORDER) + " --out " + PREPARE_OUT),
                "job": self._public(rec) if rec else None, "busy": bool(rec and rec["state"] in LIVE), "training_running": self._training_running(),
                "stage_titles": {m: [s["title"] for s in C.plan_stages(list(C.LANG_ORDER), m, False)] for m in C.MODES},
                "note": "Only public text is downloaded here. Nothing of yours is read or uploaded."}

    # ---------------------------------------------------------------- start / stop
    def _validate(self, body):
        langs = body.get("languages")
        if not isinstance(langs, list) or not all(isinstance(x, str) for x in langs):
            raise CorpusProblem("choose at least one language")
        try:
            langs = C.check_langs(langs)
        except C.CorpusError as e:
            raise CorpusProblem(str(e))
        mode = body.get("mode", "all")
        if mode not in C.MODES:
            raise CorpusProblem("mode must be one of: %s" % ", ".join(C.MODES))
        mix = body.get("mix", False)
        if not isinstance(mix, bool):
            raise CorpusProblem("mix must be true or false")
        raw_mb = body.get("mb") or {}
        if not isinstance(raw_mb, dict):
            raise CorpusProblem("mb must be an object like {\"en\": 420}")
        mb = {}
        for k, v in raw_mb.items():
            if k not in langs:
                continue
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= C.MAX_MB:
                raise CorpusProblem("the size for %s must be a whole number of MB between 0 and %d" % (k, C.MAX_MB))
            if C.LANGS[k]["capped"]:
                mb[k] = v
        return langs, mode, mix, mb

    def start(self, body):
        if body.get("confirm") is not True:
            raise CorpusProblem("confirmation missing: send confirm = true (the page asks you first)", 400)
        langs, mode, mix, mb = self._validate(body)
        if "taglish" in langs and body.get("ack_taglish") is not True:
            raise CorpusProblem("Taglish needs your acknowledgement: its posts come from real forum users, its license is unknown, "
                                "so it is for your own private training only (send ack_taglish = true).", 400)
        rec = self._record()
        if rec and rec["state"] in LIVE:
            raise CorpusProblem("a language-data job is already running. Stop it first.", 409)
        if mode == "all" and importlib.util.find_spec("datasets") is None:
            raise CorpusProblem("The `datasets` package is not installed. Run in the terminal:  pip install datasets", 409)
        if mode == "clean":
            missing = []
            for l in langs:
                spec = C.LANGS[l]
                if not self._file_info(spec["raw"] or spec["clean"])["exists"]:
                    missing.append(spec["label"])
            if missing:
                raise CorpusProblem("nothing to clean yet for: %s. Download first." % ", ".join(missing), 409)
        else:
            need = sum(((mb.get(l, C.LANGS[l]["default_mb"]) or C.LANGS[l]["default_mb"] or UNKNOWN_MB)) for l in langs) * DISK_FACTOR
            try:
                free = shutil.disk_usage(self.guard.root).free / 1e6
            except OSError:
                free = None
            if free is not None and free < need:
                raise CorpusProblem("not enough free disk space: about %d MB needed, %d MB free." % (need, free), 409)
        os.makedirs(self._path("public_text"), exist_ok=True)
        jobfile, stopfile, logfile = self._path(JOBFILE), self._path(STOPFILE), self._path(LOGFILE)
        _drop(stopfile)
        _drop(self._path(PROGRESS))
        argv = ["corpus", "--lang", ",".join(langs), "--mode", mode]
        if mb:
            argv += ["--mb"] + ["%s=%d" % (k, v) for k, v in mb.items()]
        if mix:
            argv.append("--mix")
        argv += ["--stopfile", STOPFILE, "--progress", PROGRESS]
        now = time.time()
        shown = self.terminal_command(langs, mode, mb, mix)
        rec = {"state": "starting", "step": "corpus", "langs": langs, "mode": mode, "mix": mix, "log": LOGFILE, "command": shown,
               "stages": [s["title"] for s in C.plan_stages(langs, mode, mix)], "started": now, "beat": now}
        with open(jobfile, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        with open(logfile, "wb") as lf:
            lf.write(("[web UI] %s\n" % shown).encode("utf-8", "replace"))
        env = dict(os.environ)
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kw["start_new_session"] = True
        lf = open(logfile, "ab")
        try:
            self.proc = subprocess.Popen(self.wrapper + [jobfile, stopfile, "--"] + argv, cwd=self.guard.root, env=env, stdin=subprocess.DEVNULL,
                                         stdout=lf, stderr=subprocess.STDOUT, **kw)
        except OSError as e:
            rec.update(state="failed", exit_code=-1, ended=time.time())
            with open(jobfile, "w", encoding="utf-8") as f:
                json.dump(rec, f)
            raise CorpusProblem("could not start the process: " + self._clean(e), 500)
        finally:
            lf.close()
        return {"started": True, "job": self._public({**rec, "state": "starting"})}

    def stop(self, body):
        rec = self._record()
        if not rec or rec["state"] not in LIVE:
            raise CorpusProblem("no language-data job is running", 409)
        jobfile, stopfile = self._path(JOBFILE), self._path(STOPFILE)
        if body.get("force") is True:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
            elif rec.get("pid"):
                try:
                    os.kill(int(rec["pid"]), signal.SIGTERM)
                except (OSError, ValueError):
                    pass
            full = self._load(jobfile) or {}
            full.update(state="stopped", ended=time.time(), exit_code=-15, forced=True)
            with open(jobfile, "w", encoding="utf-8") as f:
                json.dump(full, f)
            _drop(stopfile)
            return {"stopped": "forced", "job": self._public({**full})}
        with open(stopfile, "w") as f:
            f.write("stop\n")
        return {"stopped": "asked", "job": self._public({**rec, "state": "stopping"})}


def _drop(path):
    try:
        os.remove(path)
    except OSError:
        pass
