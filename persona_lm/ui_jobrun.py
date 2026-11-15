"""Stage U5b: the small wrapper the web UI starts as a background process (never run by hand).

  python -u -m persona_lm.ui_jobrun JOBFILE STOPFILE -- <the normal persona_lm arguments, e.g. train --run ...>

It runs the normal CLI (`persona_lm.cli.main`) and, next to it:
  * writes JOBFILE (JSON) with state running / done / failed / stopped, and a heartbeat every 2 s, so the UI can see the job
    again after the UI itself was restarted (no process-id probing, which is not safe on Windows);
  * watches STOPFILE: when the UI creates it, this process raises SIGINT in itself. The trainer treats SIGINT as "write a
    checkpoint and stop" (train.install_stop_flag), exactly like Ctrl+C in the terminal. Works the same on Windows and Linux.
"""
import json
import os
import signal
import sys
import threading
import time
import traceback

BEAT_SECONDS = 2.0
POLL_SECONDS = 0.25


def _write(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp, path)


def run(jobfile, stopfile, argv, main=None):
    """Returns the exit code. `main` is injectable for tests."""
    if main is None:
        from .cli import main
    rec = {"state": "running", "pid": os.getpid(), "started": time.time(), "beat": time.time(), "argv": list(argv)}
    try:
        with open(jobfile, encoding="utf-8") as f:                 # keep what the UI wrote (step, run, log, ...)
            rec = {**json.load(f), **rec}
    except (OSError, ValueError):
        pass
    _write(jobfile, rec)
    done = threading.Event()
    stop = {"asked": False}

    def watch():
        last = 0.0
        while not done.wait(POLL_SECONDS):
            if time.time() - last >= BEAT_SECONDS:
                last = rec["beat"] = time.time()
                try:
                    _write(jobfile, rec)
                except OSError:
                    pass
            if not stop["asked"] and os.path.exists(stopfile):
                stop["asked"] = True
                rec["stop_asked"] = time.time()
                try:
                    signal.raise_signal(signal.SIGINT)
                except (OSError, ValueError):
                    pass

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    code = 0
    try:
        main(list(argv))
    except KeyboardInterrupt:
        code = 130
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if isinstance(e.code, str):
            print(e.code)
    except BaseException:
        traceback.print_exc()
        code = 1
    done.set()
    t.join(timeout=3)
    rec.update(beat=time.time(), ended=time.time(), exit_code=code,
               state="stopped" if (stop["asked"] or code == 130) else "done" if code == 0 else "failed")
    try:
        os.remove(stopfile)
    except OSError:
        pass
    _write(jobfile, rec)
    return code


def _cli():
    a = sys.argv[1:]
    if len(a) < 3 or a[2] != "--":
        sys.exit("internal tool of the web UI; not meant to be run by hand")
    return run(a[0], a[1], a[3:])


if __name__ == "__main__":
    sys.exit(_cli())
