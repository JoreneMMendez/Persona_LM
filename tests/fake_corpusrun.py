"""Test helper for tests/test_ui_corpus.py: stands in for `python -m persona_lm.ui_jobrun` running `corpus ...`. Same arguments: JOBFILE STOPFILE -- ARGV...
It prints the arguments it got (so a test can check that nothing from the page reached them), writes the --progress file like the real command,
and ends cleanly on SIGINT. Behaviour: a language list containing ceb fails; an en size of 777 MB keeps going until stopped; everything else takes about 1 s."""
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import corpus, ui_jobrun


def fake_main(argv):
    assert argv[0] == "corpus", argv
    a = argv[1:]
    get = lambda name, d=None: a[a.index(name) + 1] if name in a else d
    langs = get("--lang").split(",")
    progress = get("--progress")
    stages = corpus.plan_stages(langs, get("--mode", "all"), "--mix" in a)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    print("fake corpus started: " + " ".join(argv))
    if "ceb" in langs:
        raise SystemExit("boom: download failed")
    endless = "en=777" in a
    for i, st in enumerate(stages, 1):
        corpus._write_progress(progress, i, len(stages), st["title"], st["lang"])
        print("[STEP %d/%d] %s" % (i, len(stages), st["title"]))
        for _ in range(10 if not endless else 10 ** 6):
            if stop["flag"]:
                print("stopped")
                raise KeyboardInterrupt()
            time.sleep(0.1 if not endless else 0.05)
            if not endless:
                break
    print("FINISHED")


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(ui_jobrun.run(a[0], a[1], a[3:], main=fake_main))
