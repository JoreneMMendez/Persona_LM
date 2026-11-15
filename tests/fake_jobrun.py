"""Test helper for tests/test_ui_u5b.py: stands in for `python -m persona_lm.ui_jobrun`. Same arguments: JOBFILE STOPFILE -- ARGV...
The 'trainer' writes a metrics.jsonl line every 0.2 s and, like the real one, ends cleanly when it gets SIGINT.
Mode comes from the first ARGV word: train / prepare / fail / sleep."""
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import ui_jobrun


def fake_main(argv):
    mode = argv[0]
    if mode == "fail":
        raise SystemExit("boom: bad data")
    run = argv[argv.index("--run") + 1] if "--run" in argv else None
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    print("fake %s started: %s" % (mode, " ".join(argv)))
    n = 0
    while not stop["flag"] and n < (3 if mode == "prepare" else 600):
        n += 1
        if run:
            os.makedirs(run, exist_ok=True)
            with open(os.path.join(run, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": n * 10, "loss": 2.0 / n, "lr": 1e-4, **({"val_loss": 1.0 + 0.05 * n} if n % 3 == 0 else {})}) + "\n")
        print("step %d" % (n * 10))
        time.sleep(0.2)
    print("saved checkpoint (stopped early)" if stop["flag"] else "finished")


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(ui_jobrun.run(a[0], a[1], a[3:], main=fake_main))
