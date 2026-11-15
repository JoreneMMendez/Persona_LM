"""Stage U8: Language data (persona_lm/corpus.py, persona_lm/ui_corpus.py, the corpus command, the page's routes).
No network is used. The downloaders (which need the `datasets` package) are replaced by small stub scripts in a temporary project folder;
the cleaning scripts (clean_corpus.py, clean_ceb.py, make_mix.py) are the REAL ones, copied from the project, so the cleaning rules are tested too.
The web job is run by tests/fake_corpusrun.py, which takes the same arguments as `python -m persona_lm.ui_jobrun`."""
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from persona_lm import cli
from persona_lm import corpus as C
from persona_lm import pipeline as P
from persona_lm import ui_corpus as UC
from persona_lm import ui_server as UI
from test_ui_u0 import SECRET, Base

FAKE = os.path.join(HERE, "fake_corpusrun.py")
SCRIPTS = ("grab_english", "grab_data", "ceb_domains", "grab_taglish", "clean_corpus", "clean_ceb", "make_mix")


def doc(tag, n=400):
    return ("%s " % tag) + ("kumusta ka na, ito ay isang halimbawa ng teksto para sa pagsubok. " * 20)[:n]


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


STUBS = {
    # English: writes about max_chars of text unless SHORT_EN exists (a network cut-off) or FAIL_NET exists
    "grab_english.py": '''
import json, os
def stream_into(f, repo, config, max_chars, label, already=0):
    if os.path.exists("FAIL_NET"):
        raise RuntimeError("network down")
    n = 0; chars = 0
    limit = 400 if os.path.exists("SHORT_EN") else max_chars
    while chars < limit:
        t = ("english %d " % n) + "the quick brown fox jumps over the lazy dog. " * 40
        f.write(json.dumps({"text": t}) + "\\n"); chars += len(t); n += 1
    return n, chars
''',
    "grab_data.py": '''
import json, os
def grab(name, out_path, max_chars):
    if os.path.exists("NO_DOWNLOAD"):
        raise AssertionError("download must not run in clean mode")
    if os.path.exists("FAIL_NET"):
        raise RuntimeError("network down")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps({"text": ("tagalog %d " % i) + "kumusta ka na ito ay halimbawa. " * 20}) + "\\n")
        f.write(json.dumps({"text": "too short"}) + "\\n")
        for i in range(5):
            f.write(json.dumps({"text": "SAME OPENING LINE FOR EVERY PAGE " + "x" * 400 + str(i)}) + "\\n")
    if os.path.exists("STOP_DURING"):
        open("public_text/corpus_job.stop", "w").write("stop")
''',
    "ceb_domains.py": '''
import json, os
OUT = "public_text/ceb/ceb_url.jsonl"
def main():
    if os.path.exists("NO_DOWNLOAD"):
        raise AssertionError("download must not run in clean mode")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({"text": ("cebuano %d " % i) + "unsa man nimo kaayo gyud nako dili. " * 20, "url": "https://balita.example.ph/%d" % i}) + "\\n")
        f.write(json.dumps({"text": "machine translated " + "x " * 300, "url": "https://spam.example.com/a"}) + "\\n")
''',
    "grab_taglish.py": '''
import json, os
OUT_PATH = "public_text/taglish/pex.jsonl"
def main(max_mb):
    if os.path.exists("NO_DOWNLOAD"):
        raise AssertionError("download must not run in clean mode")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "taglish 1 tara na guys, email me at juan.cruz@example.com or text 0917 123 4567. see http://forum.example.com/t/1 @kuya_jun " + "sige lang po. " * 20}) + "\\n")
        f.write(json.dumps({"text": "taglish 2 " + "grabe naman yung traffic kanina sa edsa. " * 10}) + "\\n")
        f.write(json.dumps({"text": "taglish 2 " + "grabe naman yung traffic kanina sa edsa. " * 10}) + "\\n")
        f.write(json.dumps({"text": "short"}) + "\\n")
''',
}


class PipelineBase(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.old_path = list(sys.path)
        self.tmp = tempfile.mkdtemp()
        for name in SCRIPTS:
            sys.modules.pop(name, None)
        for name, src in STUBS.items():
            with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as f:
                f.write(src)
        for name in ("clean_corpus.py", "clean_ceb.py", "make_mix.py"):          # the REAL cleaning scripts
            shutil.copy(os.path.join(ROOT, name), os.path.join(self.tmp, name))
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.old_cwd)
        sys.path[:] = self.old_path
        for name in SCRIPTS:
            sys.modules.pop(name, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def flag(self, name):
        open(os.path.join(self.tmp, name), "w").close()

    def run_c(self, langs, **kw):
        kw.setdefault("mb", {"en": 1})
        return C.run(langs, **kw)


class Planning(unittest.TestCase):
    def test_stage_order_and_titles(self):
        st = C.plan_stages(["taglish", "en"], "all", True)                        # canonical order, not the caller's
        self.assertEqual([(s["kind"], s["lang"]) for s in st],
                         [("download", "en"), ("clean", "en"), ("download", "taglish"), ("clean", "taglish"), ("mix", None), ("check", None)])
        self.assertEqual([s["kind"] for s in C.plan_stages(["tl"], "clean", False)], ["clean", "check"])

    def test_bad_input_is_refused(self):
        for bad in ([], ["fr"], ["en", "fr"], None):
            with self.assertRaises(C.CorpusError, msg=bad):
                C.plan_stages(bad)
        with self.assertRaises(C.CorpusError):
            C.plan_stages(["en"], "everything")
        for bad in (["en"], ["xx=5"], ["en=abc"], ["en=-1"], ["en=999999"]):
            with self.assertRaises(C.CorpusError, msg=bad):
                C.parse_mb(bad)
        self.assertEqual(C.parse_mb(["en=200", "tl=0"]), {"en": 200, "tl": 0})

    def test_four_languages_are_defined(self):
        self.assertEqual(C.LANG_ORDER, ("en", "tl", "ceb", "taglish"))
        for l in C.LANG_ORDER:
            self.assertTrue(C.LANGS[l]["clean"].startswith("public_clean/"))
        self.assertIn("private", C.LANGS["taglish"]["license"])

    def test_cli_command_exists(self):
        a = cli.build_parser().parse_args(["corpus", "--lang", "tl,ceb", "--mode", "clean", "--mb", "tl=10", "--mix"])
        self.assertEqual((a.lang, a.mode, a.mb, a.mix), ("tl,ceb", "clean", ["tl=10"], True))
        with self.assertRaises(SystemExit):
            cli.main(["corpus", "--lang", "klingon"])


class Pipeline(PipelineBase):
    def test_full_run_all_four_with_mix(self):
        prog = os.path.join(self.tmp, "prog.json")
        self.run_c(list(C.LANG_ORDER), mix=True, progress=prog)
        for l in C.LANG_ORDER:
            self.assertTrue(lines(C.LANGS[l]["clean"]), l)
            self.assertTrue(lines("public_mix/%s.jsonl" % l), l)
        self.assertEqual(len(lines("public_clean/tl/tl.jsonl")), 6)              # short page and the 5 repeated openings are gone
        self.assertEqual(len(lines("public_clean/ceb/ceb.jsonl")), 4)            # the non-Philippine site is gone
        left = [os.path.join(r, n) for r, _, ns in os.walk(".") for n in ns + [] if n.endswith(".part")]
        self.assertEqual(left, [], "no .part files may be left after a finished run")
        self.assertFalse(os.path.exists("public_mix/.part"))
        with open(prog) as f:
            p = json.load(f)
        self.assertEqual((p["index"], p["total"], p["title"]), (10, 10, "Finished"))

    def test_taglish_is_scrubbed_deduplicated_and_cut(self):
        self.run_c(["taglish"])
        rows = lines("public_clean/taglish/taglish.jsonl")
        self.assertEqual(len(rows), 2)                                           # short thread dropped, exact duplicate dropped
        text = " ".join(r["text"] for r in rows)
        for leak in ("juan.cruz@example.com", "0917 123 4567", "forum.example.com", "@kuya_jun"):
            self.assertNotIn(leak, text)
        for token in ("[EMAIL]", "[PHONE]", "[URL]", "[HANDLE]"):
            self.assertIn(token, text)
        self.assertIn("grabe naman yung traffic", text)                           # ordinary words are untouched

    def test_scrub_leaves_dates_and_short_numbers(self):
        out = C.scrub_public("sa 2024-05-17 may 12 tao, pm me at 09171234567")
        self.assertIn("2024-05-17", out)
        self.assertIn("12 tao", out)
        self.assertIn("[PHONE]", out)

    def test_clean_only_never_downloads(self):
        self.run_c(list(C.LANG_ORDER))                                           # first a normal run to make raw files
        shutil.rmtree("public_clean/tl")
        self.flag("NO_DOWNLOAD")
        self.run_c(["tl", "ceb", "taglish"], mode="clean")                       # would raise AssertionError if a downloader ran
        self.assertEqual(len(lines("public_clean/tl/tl.jsonl")), 6)

    def test_clean_only_without_raw_says_so(self):
        with self.assertRaises(C.CorpusError) as cm:
            self.run_c(["tl"], mode="clean")
        self.assertIn("not been downloaded", str(cm.exception))

    def test_failed_download_keeps_the_old_file(self):
        write_jsonl("public_text/tl/tl.jsonl", [{"text": "OLD GOOD FILE"}])
        self.flag("FAIL_NET")
        with self.assertRaises(RuntimeError):
            self.run_c(["tl"])
        self.assertEqual(lines("public_text/tl/tl.jsonl"), [{"text": "OLD GOOD FILE"}])

    def test_english_cut_off_does_not_replace_a_good_file(self):
        write_jsonl("public_clean/en/en.jsonl", [{"text": "OLD GOOD ENGLISH " * 30}])
        self.flag("SHORT_EN")
        with self.assertRaises(C.CorpusError) as cm:
            self.run_c(["en"])
        self.assertIn("ended early", str(cm.exception))
        self.assertEqual(lines("public_clean/en/en.jsonl"), [{"text": "OLD GOOD ENGLISH " * 30}])

    def test_stop_before_a_step_runs_nothing(self):
        stop = os.path.join(self.tmp, "stop")
        open(stop, "w").close()
        with self.assertRaises(KeyboardInterrupt):
            self.run_c(["tl"], stopfile=stop)
        self.assertFalse(os.path.exists("public_text"))

    def test_stop_during_a_download_does_not_use_the_partial_file(self):
        os.makedirs("public_text", exist_ok=True)
        self.flag("STOP_DURING")                                                 # the stub asks for a stop right after writing
        with self.assertRaises(KeyboardInterrupt):
            self.run_c(["tl"], stopfile=os.path.join(self.tmp, "public_text", "corpus_job.stop"))
        self.assertFalse(os.path.exists("public_text/tl/tl.jsonl"))
        self.assertFalse(os.path.exists("public_clean/tl/tl.jsonl"))

    def test_a_cleaning_that_keeps_nothing_leaves_the_old_clean_file(self):
        write_jsonl("public_clean/tl/tl.jsonl", [{"text": "OLD CLEAN " * 50}])
        write_jsonl("public_text/tl/tl.jsonl", [{"text": "short"}])
        with self.assertRaises(C.CorpusError):
            self.run_c(["tl"], mode="clean")
        self.assertEqual(lines("public_clean/tl/tl.jsonl"), [{"text": "OLD CLEAN " * 50}])
        self.assertFalse(os.path.exists("public_clean/tl/tl.jsonl.part"))

    def test_check_finds_unreadable_lines(self):
        os.makedirs("public_clean/en", exist_ok=True)
        with open("public_clean/en/en.jsonl", "w") as f:
            f.write(json.dumps({"text": "fine " * 100}) + "\n{broken json\n")
        with self.assertRaises(C.CorpusError) as cm:
            self.run_c(["en"], mode="clean")
        self.assertIn("unreadable", str(cm.exception))

    def test_missing_datasets_package_gives_the_install_command(self):
        os.remove("grab_data.py")
        with open("grab_data.py", "w") as f:
            f.write("from datasets import load_dataset\n")                       # what the real script does at import
        with mock.patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaises(C.CorpusError) as cm:
                self.run_c(["tl"])
        self.assertIn("pip install datasets", str(cm.exception))

    def test_output_feeds_prepare(self):
        """The mix files are what `prepare --kind pretrain --data` reads: {"text": ...} per line."""
        from persona_lm.data import _json_records
        self.run_c(["en", "tl"], mix=True)
        recs = list(_json_records("public_mix/tl.jsonl"))
        self.assertTrue(recs and all(isinstance(r["text"], str) for r in recs))


class RoutesBase(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.state.corpus.wrapper = [sys.executable, "-u", FAKE]
        cls.pt = os.path.join(cls.root, "public_text")

    def tearDown(self):
        try:
            self.api("POST", "/api/corpus/stop", {"force": True})
        except Exception:
            pass
        p = self.state.corpus.proc
        if p is not None:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        shutil.rmtree(self.pt, ignore_errors=True)
        for d in ("public_clean", "public_mix"):
            shutil.rmtree(os.path.join(self.root, d), ignore_errors=True)

    def get(self, path, **kw):
        st, _, body = self.api("GET", path, **kw)
        return st, json.loads(body or b"{}")

    def post(self, path, body, **kw):
        st, _, data = self.api("POST", path, body, **kw)
        return st, json.loads(data or b"{}")

    def start(self, **over):
        body = {"languages": ["tl"], "mode": "all", "confirm": True}
        body.update(over)
        with mock.patch.object(UC.importlib.util, "find_spec", lambda n, *a: object() if n == "datasets" else importlib.util.find_spec(n, *a)):
            return self.post("/api/corpus/start", body)

    def wait(self, pred, what, timeout=20):
        end = time.time() + timeout
        j = {}
        while time.time() < end:
            st, j = self.get("/api/corpus/job")
            self.assertEqual(st, 200, j)
            if pred(j):
                return j
            time.sleep(0.15)
        self.fail("timed out waiting for " + what + ": " + json.dumps(j)[:600])


class Safety(RoutesBase):
    def test_listed_and_guarded(self):
        for p in ("/api/corpus/overview", "/api/corpus/job"):
            self.assertIn(p, UI.ROUTES["GET"])
            self.assertNotIn(p, UI.ROUTES["POST"])
        self.assertEqual([p for p in UI.ROUTES["POST"] if "/corpus" in p], ["/api/corpus/start", "/api/corpus/stop"])
        self.assertIn("/static/corpus.js", UI.ROUTES["GET"])
        for m, p, body in (("GET", "/api/corpus/overview", None), ("GET", "/api/corpus/job", None),
                           ("POST", "/api/corpus/start", {"languages": ["tl"], "confirm": True}), ("POST", "/api/corpus/stop", {})):
            self.assertEqual(self.api(m, p, body, token=False)[0], 401, p)
            self.assertEqual(self.api(m, p, body, token="wrong")[0], 401, p)
            self.assertEqual(self.api(m, p, body, host="evil.example.com")[0], 403, p)
            self.assertEqual(self.api(m, p, body, headers={"Origin": "http://evil.example.com"})[0], 403, p)
        self.assertFalse(os.path.exists(os.path.join(self.pt, "corpus_job.json")))        # nothing started by the refused calls

    def test_wrong_method_and_content_type(self):
        self.assertEqual(self.api("GET", "/api/corpus/start")[0], 404)
        self.assertEqual(self.api("POST", "/api/corpus/job", {})[0], 404)
        self.assertEqual(self.api("POST", "/api/corpus/start", "languages=tl", headers={"Content-Type": "text/plain"})[0], 415)

    def test_the_page_cannot_choose_the_command(self):
        st, j = self.start(languages=["tl", "en"], mb={"tl": 5}, mix=True, argv=["--evil"], cmd="del *", path="../x", url="http://evil.example.com/x",
                           dataset="evil/data", out="/tmp/x", stopfile="/etc/passwd")
        self.assertEqual(st, 200, j)
        done = self.wait(lambda j: j["job"] and j["job"]["state"] == "done", "the job to finish")["job"]
        log = "\n".join(done["log"])
        self.assertIn("--lang en,tl --mode all --mb tl=5 --mix --stopfile public_text/corpus_job.stop", log)
        for bad in ("evil", "passwd", "del *", "/tmp/x", "../x"):
            self.assertNotIn(bad, log)
        self.assertEqual(done["command"], "python -m persona_lm corpus --lang en,tl --mode all --mb tl=5 --mix")

    def test_bad_requests_are_refused_and_start_nothing(self):
        bad = [{"confirm": None}, {"confirm": "yes"}, {"languages": []}, {"languages": "tl"}, {"languages": ["fr"]}, {"languages": [1]},
               {"mode": "wipe"}, {"mix": "yes"}, {"mb": [1]}, {"mb": {"tl": -1}}, {"mb": {"tl": "5"}}, {"mb": {"tl": True}}, {"mb": {"tl": 1.5}}, {"mb": {"tl": 10 ** 9}}]
        for over in bad:
            st, j = self.start(**over)
            self.assertEqual(st, 400, (over, j))
            self.assertFalse(j["ok"])
        self.assertFalse(os.path.exists(os.path.join(self.pt, "corpus_job.json")))

    def test_taglish_needs_the_acknowledgement(self):
        st, j = self.start(languages=["taglish"])
        self.assertEqual(st, 400)
        self.assertIn("private", j["error"])
        st, j = self.start(languages=["taglish"], ack_taglish=True)
        self.assertEqual(st, 200, j)

    def test_download_needs_the_datasets_package(self):
        with mock.patch.object(UC.importlib.util, "find_spec", lambda n, *a: None if n == "datasets" else importlib.util.find_spec(n, *a)):
            st, j = self.post("/api/corpus/start", {"languages": ["tl"], "mode": "all", "confirm": True})
            self.assertEqual(st, 409)
            self.assertIn("pip install datasets", j["error"])
            self.assertFalse(self.get("/api/corpus/overview")[1]["datasets_installed"])

    def test_clean_only_needs_something_downloaded(self):
        st, j = self.start(languages=["tl", "ceb"], mode="clean")
        self.assertEqual(st, 409)
        self.assertIn("Tagalog", j["error"])

    def test_not_enough_disk_space(self):
        with mock.patch.object(UC.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10 ** 9, 9 * 10 ** 8, 10 ** 6)):
            st, j = self.start(languages=["en"], mb={"en": 400})
        self.assertEqual(st, 409)
        self.assertIn("disk space", j["error"])

    def test_writes_only_inside_the_public_text_folders(self):
        before = set(os.listdir(self.root))
        self.start(languages=["tl"])
        self.wait(lambda j: j["job"] and j["job"]["state"] == "done", "done")
        self.assertEqual(set(os.listdir(self.root)) - before, {"public_text"})
        for n in ("corpus_job.json", "corpus.log", "corpus_progress.json"):
            self.assertTrue(os.path.exists(os.path.join(self.pt, n)), n)

    def test_private_folders_stay_blocked_and_public_ones_are_allowed(self):
        g = self.state.guard
        for ok in ("public_text", "public_clean/en/en.jsonl", "public_mix/tl.jsonl"):
            g.resolve(ok)
        for bad in ("data_msgr/private/x", "public_text/private/x", "public_text/../memory.json", "/etc/passwd", "C:\\x", "raw/x"):
            with self.assertRaises(UI.UIProblem, msg=bad):
                g.resolve(bad)

    def test_no_text_of_the_files_reaches_the_page(self):
        os.makedirs(os.path.join(self.root, "public_clean", "en"))
        with open(os.path.join(self.root, "public_clean", "en", "en.jsonl"), "w") as f:
            f.write(json.dumps({"text": SECRET}) + "\n")
        st, _, body = self.api("GET", "/api/corpus/overview")
        self.assertEqual(st, 200)
        self.assertNotIn(SECRET.encode(), body)

    def test_privacy_page_tells_the_truth_about_the_download(self):
        _, p = self.get("/api/privacy")
        self.assertTrue(any("Language data" in t for t in p["never_leaves"]))
        paths = {i["path"] for i in p["stored"]}
        self.assertTrue({"public_text", "public_clean", "public_mix"} <= paths)
        sec = [s for s in self.get("/api/info")[1]["sections"] if s["id"] == "corpus"][0]
        self.assertTrue(sec["ready"])
        self.assertEqual(sec["label"], "Language data")

    def test_the_page_script_is_served_and_has_no_unsafe_calls(self):
        st, hd, body = self.api("GET", "/static/corpus.js")
        self.assertEqual(st, 200)
        self.assertIn("javascript", hd["content-type"])
        js = body.decode("utf-8")
        self.assertNotRegex(js, r"\.innerHTML\s*=|insertAdjacentHTML|document\.write|\beval\(|new Function")
        self.assertNotRegex(js, r"https?://")


class Overview(RoutesBase):
    def test_idle_shape(self):
        st, o = self.get("/api/corpus/overview")
        self.assertEqual(st, 200)
        self.assertEqual([l["id"] for l in o["languages"]], ["en", "tl", "ceb", "taglish"])
        for l in o["languages"]:
            self.assertEqual(l["state"], "missing")
            self.assertFalse(l["clean"]["exists"])
        self.assertIsNone(o["job"])
        self.assertFalse(o["busy"])
        self.assertIsInstance(o["datasets_installed"], bool)
        self.assertEqual(o["install_command"], "pip install datasets")
        self.assertTrue([l for l in o["languages"] if l["id"] == "taglish"][0]["needs_ack"])

    def test_states_follow_the_files(self):
        write_jsonl(os.path.join(self.root, "public_clean/en/en.jsonl"), [{"text": "x" * 500}])
        write_jsonl(os.path.join(self.root, "public_text/tl/tl.jsonl"), [{"text": "x" * 500}])
        write_jsonl(os.path.join(self.root, "public_text/ceb/ceb_url.jsonl.part"), [{"text": "x"}])
        by = {l["id"]: l for l in self.get("/api/corpus/overview")[1]["languages"]}
        self.assertEqual((by["en"]["state"], by["tl"]["state"], by["ceb"]["state"], by["taglish"]["state"]), ("ready", "downloaded", "missing", "missing"))
        self.assertTrue(by["ceb"]["partial"])
        self.assertGreater(by["en"]["clean"]["modified"], 0)

    def test_what_each_project_reads(self):
        work = os.path.join(self.root, "work")
        P.save_project(work, {**P.DEFAULTS, "chat_train": "data_msgr/train.jsonl", "pretrain_text": ["public_text/", "public_mix\\en.jsonl", "elsewhere/x.txt"]})
        try:
            projs = self.get("/api/corpus/overview")[1]["projects"]
            w = [p for p in projs if p["workdir"] == "work"][0]
            self.assertEqual([r["kind"] for r in w["reads"]], ["public_text", "public_mix", "other"])
            self.assertFalse(w["tokens_done"])
        finally:
            os.remove(P.project_path(work))

    def test_training_running_is_reported(self):
        with mock.patch.object(self.state.job, "_current", lambda: {"state": "running", "workdir": "work"}):
            self.assertTrue(self.get("/api/corpus/overview")[1]["training_running"])
            self.assertTrue(self.get("/api/corpus/job")[1]["training_running"])
        self.assertFalse(self.get("/api/corpus/job")[1]["training_running"])


class Jobs(RoutesBase):
    def test_a_run_goes_to_done_and_shows_steps(self):
        st, j = self.start(languages=["tl", "en"], mix=True, mb={"en": 5})
        self.assertEqual(st, 200, j)
        self.assertTrue(j["started"])
        self.assertIn(j["job"]["state"], ("starting", "running"))
        self.assertEqual(len(j["job"]["stages"]), 6)
        d = self.wait(lambda j: j["job"] and j["job"]["state"] == "done", "done")
        self.assertEqual(d["job"]["step"]["total"], 6)
        self.assertEqual(d["job"]["step"]["index"], 6)
        self.assertIn("FINISHED", d["job"]["log"])
        self.assertFalse(d["busy"])

    def test_a_failure_is_reported_with_its_log(self):
        self.start(languages=["ceb"])
        d = self.wait(lambda j: j["job"] and j["job"]["state"] == "failed", "failed")["job"]
        self.assertEqual(d["exit_code"], 1)
        self.assertTrue(any("boom" in x for x in d["log"]))

    def test_one_job_at_a_time_and_stop(self):
        st, j = self.start(languages=["en"], mb={"en": 777})
        self.assertEqual(st, 200, j)
        self.wait(lambda j: j["job"] and j["job"]["state"] == "running" and j["job"]["step"]["index"], "running")
        self.assertTrue(self.get("/api/corpus/job")[1]["busy"])
        st, j2 = self.start(languages=["tl"])
        self.assertEqual(st, 409)
        self.assertIn("already running", j2["error"])
        st, s = self.post("/api/corpus/stop", {})
        self.assertEqual((st, s["stopped"]), (200, "asked"))
        end = self.wait(lambda j: j["job"] and j["job"]["state"] == "stopped", "stopped")["job"]
        self.assertEqual(end["exit_code"], 130)
        self.assertFalse(os.path.exists(os.path.join(self.pt, "corpus_job.stop")))        # the runner removes the stop file when it ends
        self.assertEqual(self.post("/api/corpus/stop", {})[0], 409)                        # nothing left to stop
        self.assertEqual(self.start(languages=["tl"])[0], 200)                             # and a new one can start

    def test_force_stop_ends_the_process(self):
        self.start(languages=["en"], mb={"en": 777})
        self.wait(lambda j: j["job"] and j["job"]["state"] == "running", "running")
        st, s = self.post("/api/corpus/stop", {"force": True})
        self.assertEqual((st, s["stopped"]), (200, "forced"))
        self.wait(lambda j: j["job"] and j["job"]["state"] == "stopped", "stopped")
        self.assertFalse(self.get("/api/corpus/job")[1]["busy"])

    def test_a_job_with_no_heartbeat_is_lost_not_running_forever(self):
        os.makedirs(self.pt, exist_ok=True)
        with open(os.path.join(self.pt, "corpus_job.json"), "w") as f:
            json.dump({"state": "running", "langs": ["tl"], "mode": "all", "beat": time.time() - 1000, "started": time.time() - 1000, "stages": []}, f)
        j = self.get("/api/corpus/job")[1]
        self.assertEqual(j["job"]["state"], "lost")
        self.assertFalse(j["busy"])
        self.assertEqual(self.start(languages=["tl"])[0], 200)                             # a lost job does not block a new one


if __name__ == "__main__":
    unittest.main()
