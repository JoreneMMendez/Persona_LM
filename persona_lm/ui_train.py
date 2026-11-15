"""Stage U5a: the Training page routes of the web UI (read only).

  GET /api/train/plan?workdir=work          the pipeline plan (same steps as `pipeline plan`): done / NEXT / todo / skipped + the exact command
  GET /api/train/checkpoints                every run folder under work, work2, work3 that has a checkpoint: step, size, tokenizer match, best/last val loss
  GET /api/train/history?run=work3/run_chat2   numbers of one run's metrics.jsonl (loss and val loss), for the tables on the page
  GET /api/train/log?run=...&file=          the run's log as text: built from metrics.jsonl, or the tail of a *.log file that sits in the run folder

Rules (tests in tests/test_ui_u5a.py):
  * Nothing here writes a file or starts a program. Starting and stopping training is stage U5b.
  * Only the fixed work folders (WORKDIRS) are looked at, and only one level down (run folders). Every name goes through PathGuard.
  * Checkpoints: only the small meta block is read (report.read_ckpt_meta); the weights are never loaded and no pickle is used.
  * Logs: only metrics.jsonl and files named *.log directly inside a run folder, at most the last 64 KB. The page shows them as plain text.
  * The pipeline plan reads project.json and counts of the token folders. It does not open the chat text files named in project.json
    (those paths are only shown inside the commands, as the terminal needs them).
"""
import json
import os
import re

from . import pipeline as P
from . import report as R
from .bpe import Tokenizer

WORKDIRS = ("work", "work2", "work3")
TOKENIZER = "work/tok.json"
MAX_POINTS = 300
LOG_TAIL_BYTES = 65536
LOG_TAIL_LINES = 200
MAX_METRICS_BYTES = 50_000_000
RISE_WARN = 1.03                                    # last val loss this much above the best = "rising"
LOG_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,60}\.log$")
CFG_KEYS = ("n_layers", "d_model", "n_heads", "d_ff", "ctx", "vocab_size")


class TrainProblem(Exception):
    def __init__(self, msg, code=400):
        Exception.__init__(self, msg)
        self.code = code


class TrainHub:
    def __init__(self, guard):
        self.guard = guard

    def _clean(self, e):
        return str(e).replace(self.guard.root, ".")

    def _path(self, rel):
        try:
            return self.guard.resolve(rel)
        except Exception as e:
            raise TrainProblem(self._clean(e), 403)

    def _workdir(self, name):
        if name not in WORKDIRS:
            raise TrainProblem("unknown work folder (choose one of: %s)" % ", ".join(WORKDIRS))
        return name, self._path(name)

    def _run(self, rel):
        parts = [p for p in re.split(r"[\\/]+", rel or "") if p]
        if len(parts) != 2 or parts[0] not in WORKDIRS or parts[1] in (".", ".."):
            raise TrainProblem("a run is given as WORKFOLDER/RUNFOLDER, for example work3/run_chat2")
        full = self._path("/".join(parts))
        if not os.path.isdir(full):
            raise TrainProblem("no such run folder", 404)
        return "/".join(parts), full

    # ---------------------------------------------------------------- plan
    def plan(self, workdir="work"):
        name, full = self._workdir(workdir)
        out = {"workdir": name, "workdirs": list(WORKDIRS), "exists": False}
        if not os.path.exists(P.project_path(full)):
            out["init_command"] = ("python -m persona_lm pipeline init --workdir %s --languages ceb,tl,en --pretrain-text PATH\\public_text "
                                   "--chat-train data_msgr\\train.jsonl --chat-val data_msgr\\val.jsonl") % name
            return out
        try:
            proj = P.load_project(full)
        except Exception as e:
            out["error"] = "project.json could not be read: " + self._clean(e)
            return out
        out["exists"] = True
        out["project"] = {"size": proj["size"], "ctx": proj["ctx"], "vocab": proj["vocab"], "languages": list(proj["languages"]),
                          "decisions_confirmed": bool(proj["decisions_confirmed"]), "confirmed_at": proj.get("confirmed_at"),
                          "finetune_epochs": proj["finetune_epochs"], "finetune_lr": proj["finetune_lr"], "pretrain_steps": proj["pretrain_steps"]}
        try:
            steps = P.build_steps(proj, full)
        except Exception as e:
            out["error"] = self._clean(e)
            return out
        nxt = P.next_step(steps)
        out["steps"] = [{"name": s["name"], "title": s["title"], "note": s.get("note") or "",
                         "state": "done" if s["done"] else "skipped" if s["skipped"] else "next" if s is nxt else "todo",
                         "command": P.show(s["argv"]) if s.get("argv") else None} for s in steps]
        out["chat_command"] = P.show(P.chat_command(proj, full))
        out["next"] = nxt["name"] if nxt else None
        out["all_done"] = nxt is None
        return out

    # ---------------------------------------------------------------- checkpoints
    def _frozen_sha(self):
        try:
            return Tokenizer.load(self._path(TOKENIZER)).fingerprint()
        except Exception:
            return None

    def _metrics_path(self, run_full):
        p = os.path.join(run_full, "metrics.jsonl")
        return p if os.path.isfile(p) else None

    def _run_info(self, rel, full, frozen):
        main, prev = os.path.join(full, "ckpt.npz"), os.path.join(full, "ckpt.prev.npz")
        if not (os.path.exists(main) or os.path.exists(prev)):
            return None
        meta = R.read_ckpt_meta(main)
        info = {"run": rel, "has_main": os.path.exists(main), "has_prev": os.path.exists(prev), "readable": bool(meta)}
        keep = main if os.path.exists(main) else prev
        try:
            st = os.stat(keep)
            info.update({"size_mb": round(st.st_size / 1048576, 1), "modified": int(st.st_mtime)})
        except OSError:
            pass
        if meta:
            cfg = meta.get("cfg") or {}
            hp = meta.get("hp") or {}
            sha = meta.get("tokenizer_sha256")
            info.update({"step": meta.get("step"), "steps_planned": hp.get("steps"), "model": {k: cfg[k] for k in CFG_KEYS if k in cfg},
                         "lr": hp.get("lr"), "tokenizer": (sha or "")[:8] or None,
                         "tokenizer_ok": None if not (sha and frozen) else sha == frozen})
        mp = self._metrics_path(full)
        if mp:
            t = R.validation_trend(mp)
            if t:
                best, best_step, last, last_step = t
                info["val"] = {"best": best, "best_step": best_step, "last": last, "last_step": last_step,
                               "rising": bool(last > best * RISE_WARN and last_step > best_step)}
        return info

    def checkpoints(self):
        frozen = self._frozen_sha()
        runs = []
        for wd in WORKDIRS:
            try:
                base = self._path(wd)
            except TrainProblem:
                continue
            if not os.path.isdir(base):
                continue
            for d in sorted(os.listdir(base)):
                rel = wd + "/" + d
                try:
                    full = self._path(rel)
                except TrainProblem:
                    continue
                if os.path.isdir(full):
                    info = self._run_info(rel, full, frozen)
                    if info:
                        runs.append(info)
        return {"runs": runs, "tokenizer_present": frozen is not None, "tokenizer_sha": (frozen or "")[:8] or None,
                "note": "The trainer keeps only the newest checkpoint and the one before it (ckpt.prev.npz). A run whose val loss is rising went past its best step."}

    # ---------------------------------------------------------------- history and log
    def _read_metrics(self, run_full):
        mp = self._metrics_path(run_full)
        if not mp:
            return []
        try:
            if os.path.getsize(mp) > MAX_METRICS_BYTES:
                raise TrainProblem("metrics.jsonl is too big to show", 422)
            rows = []
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and isinstance(r.get("step"), int):
                        rows.append(r)
            return rows
        except OSError:
            return []

    @staticmethod
    def _thin(rows, n):
        if len(rows) <= n:
            return rows
        k = len(rows) / float(n)
        picked = [rows[int(i * k)] for i in range(n)]
        if picked[-1] is not rows[-1]:
            picked[-1] = rows[-1]
        return picked

    def history(self, run):
        rel, full = self._run(run)
        rows = self._read_metrics(full)
        train = [{"step": r["step"], "loss": r.get("loss"), "lr": r.get("lr")} for r in rows if isinstance(r.get("loss"), (int, float))]
        val = [{"step": r["step"], "val_loss": r["val_loss"]} for r in rows if isinstance(r.get("val_loss"), (int, float))]
        best = min(val, key=lambda r: r["val_loss"]) if val else None
        for v in val:
            v["best"] = bool(best and v is best)
        return {"run": rel, "rows": len(rows), "train": self._thin(train, MAX_POINTS), "val": val[-MAX_POINTS:], "best": best,
                "last_step": rows[-1]["step"] if rows else None, "has_metrics": bool(rows)}

    def log(self, run, file=""):
        rel, full = self._run(run)
        files = sorted(n for n in os.listdir(full) if LOG_NAME.match(n) and os.path.isfile(os.path.join(full, n)))
        if file:
            if file not in files:
                raise TrainProblem("that log file is not in the run folder", 404)
            path = os.path.join(full, file)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                if size > LOG_TAIL_BYTES:
                    f.seek(size - LOG_TAIL_BYTES)
                raw = f.read(LOG_TAIL_BYTES)
            lines = raw.decode("utf-8", "replace").splitlines()
            if size > LOG_TAIL_BYTES:
                lines = lines[1:]                                       # the first line is cut in the middle
            return {"run": rel, "files": files, "file": file, "source": "file", "lines": lines[-LOG_TAIL_LINES:], "cut": size > LOG_TAIL_BYTES or len(lines) > LOG_TAIL_LINES}
        rows = self._read_metrics(full)
        lines = []
        for r in rows:
            s = "step %d" % r["step"]
            if isinstance(r.get("loss"), (int, float)):
                s += "  loss %.4f" % r["loss"]
            if isinstance(r.get("val_loss"), (int, float)):
                s += "  val %.4f" % r["val_loss"]
            if isinstance(r.get("lr"), (int, float)):
                s += "  lr %.2e" % r["lr"]
            lines.append(s)
        return {"run": rel, "files": files, "file": "", "source": "metrics", "lines": lines[-LOG_TAIL_LINES:], "cut": len(lines) > LOG_TAIL_LINES}
