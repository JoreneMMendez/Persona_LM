"""Stage 9b tests: the export package (what is inside and what is never inside, privacy confirmation, weights-only, checksums, no overwriting,
lineage entry), loading from a package without PyTorch, and the local HTTP server (loopback, token, limits, text is never a command,
persona switching off by default). Tiny random model, loopback sockets only, no keyboard."""
import contextlib
import http.client
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona_lm import cli
from persona_lm import export_pkg as EX
from persona_lm import identity as I
from persona_lm import lineage as L
from persona_lm import memory as M
from persona_lm import personas as PS
from persona_lm import runtime as RT
from persona_lm import server as SV
from persona_lm.bpe import Tokenizer
from persona_lm.ckpt import save_ckpt
from persona_lm.gguf import read_gguf
from persona_lm.model import Config, init_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = "the cat sat on the mat. luna loves mango and the dog. hello how are you today fine thanks. maria is my sister, she loves mango. " * 40
SECRET_EXAMPLE = "my very private example sentence"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def all_files(folder):
    return sorted(os.path.relpath(os.path.join(r, n), folder).replace(os.sep, "/") for r, _, ns in os.walk(folder) for n in ns)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_all = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        r = cls.tmp_all.name
        Tokenizer.train([CORPUS], 300).save(os.path.join(r, "tok.json"))
        tok = Tokenizer.load(os.path.join(r, "tok.json"))
        cls.cfg = Config(vocab_size=tok.vocab_size, n_layer=2, d_model=64, n_head=4, d_ff=192, ctx=512)
        save_ckpt(os.path.join(r, "ck.npz"), init_params(cls.cfg, seed=21), {"m.tok": init_params(cls.cfg, seed=1)["tok"]},
                  {"cfg": cls.cfg.to_dict(), "step": 7})
        cls.r = r

    @classmethod
    def tearDownClass(cls):
        cls.tmp_all.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        for n in ("tok.json", "ck.npz"):
            with open(os.path.join(self.r, n), "rb") as a, open(self.p(n), "wb") as b:
                b.write(a.read())
        store = M.MemoryStore(self.p("mem.json"))
        store.add("Maria is my sister and loves mango", importance=0.6, nodes=["person:Maria"])
        store.save()
        write(self.p("card.md"), "# Person card\n## Name\n- Jaymar\n## Habits\n- I wake up early\n")
        self.core = I.Core.create(self.p("core"), "Luna", "Jaymar", personality="curious", values="honesty", tokenizer=self.p("tok.json"),
                                  memory=self.p("mem.json"), person=self.p("card.md"), mood=self.p("mood.json"))
        self.core.add_version("v1", self.p("ck.npz"))
        ps = PS.Personas(I.Core(self.p("core")))
        ps.add("calm mentor", "You speak slowly and kindly.")
        ps.add_example("calm mentor", "hi", SECRET_EXAMPLE)
        self.out = self.p("pkg")

    def p(self, *parts):
        return os.path.join(self.root, *parts)

    def export(self, **kw):
        kw.setdefault("version", "v1")
        kw.setdefault("confirmed", True)
        kw.setdefault("out", lambda *a: None)
        return EX.export_package(I.Core(self.p("core")), kw.pop("out_dir", self.out), **kw)


class Exporting(Base):
    def test_full_package_contents_and_manifest(self):
        m = self.export()
        files = all_files(self.out)
        for need in ("package.json", "model/ckpt.npz", "tokenizer.json", "identity/identity.json", "memory/memory.json", "memory/vault/README.md",
                     "person/person.md", "mood/mood.json", "personas/personas.json"):
            self.assertIn(need, files)
        self.assertEqual(set(files) - {"package.json"}, set(m["files"]))
        self.assertEqual(EX.verify_package(self.out), ([], []))
        self.assertTrue(m["files"]["memory/memory.json"]["mutable"])
        self.assertFalse(m["files"]["model/ckpt.npz"]["mutable"])
        self.assertEqual(m["version"], "v1")

    def test_nothing_private_beyond_the_listed_parts_is_inside(self):
        self.export()
        blob = "".join(read(os.path.join(self.out, f)) for f in all_files(self.out) if f.endswith((".json", ".md")))
        self.assertNotIn(SECRET_EXAMPLE, blob)                         # persona examples stay home
        self.assertNotIn("lineage", " ".join(f for f in all_files(self.out)))
        self.assertFalse(any("proposal" in f or "probe" in f for f in all_files(self.out)))
        ps = json.loads(read(os.path.join(self.out, "personas", "personas.json")))
        self.assertEqual(ps["personas"], {"calm mentor": {"text": "You speak slowly and kindly."}})

    def test_weights_only_has_no_identity_memory_card_mood_or_masks(self):
        m = self.export(weights_only=True)
        self.assertEqual(sorted(m["files"]), ["model/ckpt.npz", "tokenizer.json"])
        self.assertFalse(any(m["contains"].values()))

    def test_no_flags_leave_parts_out(self):
        m = self.export(memory=False, person=False, mood=False, personas=False)
        self.assertEqual(sorted(m["files"]), ["identity/identity.json", "model/ckpt.npz", "tokenizer.json"])

    def test_the_checkpoint_in_the_package_has_no_optimizer_state(self):
        from persona_lm.ckpt import load_ckpt
        self.export()
        params, opt, meta = load_ckpt(os.path.join(self.out, "model", "ckpt.npz"))
        self.assertEqual(opt, {})
        self.assertEqual(meta["cfg"], self.cfg.to_dict())

    def test_without_yes_it_only_warns_and_writes_nothing(self):
        with self.assertRaises(EX.ExportProblem) as cm:
            self.export(confirmed=False)
        self.assertIn("PRIVACY", str(cm.exception))
        self.assertIn("--yes", str(cm.exception))
        self.assertFalse(os.path.exists(self.out) or os.path.exists(self.out + ".partial"))

    def test_never_overwrites_and_cleans_up_after_failure(self):
        os.makedirs(self.out)
        write(os.path.join(self.out, "x.txt"), "mine")
        with self.assertRaises(EX.ExportProblem):
            self.export()
        self.assertEqual(os.listdir(self.out), ["x.txt"])
        with self.assertRaises(I.IdentityProblem):
            self.export(out_dir=self.p("pkg2"), version="nope")
        self.assertFalse(os.path.exists(self.p("pkg2.partial")))

    def test_changed_checkpoint_stops_the_export(self):
        with open(self.p("ck.npz"), "ab") as f:
            f.write(b"x")
        with self.assertRaises(I.IdentityProblem):
            self.export()
        self.assertFalse(os.path.exists(self.out))

    def test_export_is_logged_without_texts_and_the_chain_stays_valid(self):
        self.export()
        lin = L.Lineage(self.p("core", "lineage.jsonl"))
        self.assertTrue(lin.verify()[0])
        e = [x for x in lin.entries() if x["kind"] == "exported"]
        self.assertEqual(len(e), 1)
        self.assertNotIn("mango", json.dumps(e[0]))
        I.Core(self.p("core"))                                            # core.json still agrees with the log end

    def test_gguf_formats(self):
        m = self.export(fmt="gguf")
        self.assertIn("model/model.gguf", m["files"])
        self.assertNotIn("model/ckpt.npz", m["files"])
        meta, tensors = read_gguf(os.path.join(self.out, "model", "model.gguf"))
        self.assertTrue(len(tensors) > 0)
        with self.assertRaises(RT.RuntimeProblem):
            RT.load_package(self.out)                                      # the python runtime needs the npz
        with self.assertRaises(EX.ExportProblem):
            self.export(fmt="zip", out_dir=self.p("pkgz"))

    def test_verify_finds_damage_and_treats_use_as_normal(self):
        self.export()
        with open(os.path.join(self.out, "memory", "memory.json"), "a") as f:
            f.write(" ")
        problems, notes = EX.verify_package(self.out)
        self.assertEqual(problems, [])
        self.assertTrue(any("memory/memory.json" in n for n in notes))
        with open(os.path.join(self.out, "tokenizer.json"), "a") as f:
            f.write(" ")
        os.remove(os.path.join(self.out, "person", "person.md"))
        problems, _ = EX.verify_package(self.out)
        self.assertEqual(len(problems), 2)

    def test_cli_commands(self):
        run = lambda *argv: cli.main(list(argv))
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(io.StringIO()):
                run("export-package", "--dir", self.p("core"), "--out", self.out, "--version", "v1")
        self.assertIn("PRIVACY", str(cm.exception))
        with contextlib.redirect_stdout(io.StringIO()):
            run("export-package", "--dir", self.p("core"), "--out", self.out, "--version", "v1", "--yes")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run("verify-package", self.out)
        self.assertIn("OK", buf.getvalue())
        with open(os.path.join(self.out, "tokenizer.json"), "a") as f:
            f.write(" ")
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()):
                run("verify-package", self.out)
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()):
                run("serve")                                                # neither --package nor --core


class Loading(Base):
    def test_package_loads_and_chats_with_every_part(self):
        self.export()
        ai = RT.load_package(self.out, max_new=8, seed=1)
        self.assertEqual(ai.name, "Luna")
        self.assertEqual(ai.version, "v1")
        self.assertIsNotNone(ai.memory)
        self.assertIsNotNone(ai.mood)
        self.assertIsNotNone(ai.person)
        self.assertIsInstance(ai.chat("maria loves mango, hello"), str)
        self.assertIn("m0001", " ".join(ai.last_notes))
        self.assertEqual(ai.pending_proposals(), [])

    def test_same_replies_as_loading_the_core(self):
        self.export(mood=False)
        kw = dict(max_new=10, seed=4)
        a = RT.load_package(self.out, **kw)
        b = RT.load(self.p("core"), version="v1", **kw)
        b.session.mood = None                                              # the package was made without a mood
        for t in ("hello", "how are you today"):
            self.assertEqual(a.chat(t), b.chat(t))

    def test_persona_switch_in_a_package_is_in_memory_only(self):
        self.export()
        ai = RT.load_package(self.out, max_new=6)
        before = read(os.path.join(self.out, "personas", "personas.json"))
        self.assertIn("calm mentor", ai.switch_persona("calm mentor"))
        self.assertEqual(ai.persona, "calm mentor")
        ids, _ = (ai.chat("hello"), ai.session.build_prompt())[1]
        self.assertIn(ai.tok.encode("<|persona|>")[0], ids)
        self.assertEqual(before, read(os.path.join(self.out, "personas", "personas.json")))
        with self.assertRaises(RT.RuntimeProblem):
            ai.switch_persona("no such")
        ai.switch_persona("none")
        self.assertIsNone(ai.persona)

    def test_weights_only_package_loads(self):
        self.export(weights_only=True)
        ai = RT.load_package(self.out, system="You are Luna.", max_new=6)
        self.assertIsNone(ai.memory)
        self.assertIsNone(ai.mood)
        self.assertIsNone(ai.personas)
        self.assertIsInstance(ai.chat("hello"), str)

    def test_damaged_package_is_refused(self):
        self.export()
        with open(os.path.join(self.out, "model", "ckpt.npz"), "ab") as f:
            f.write(b"x")
        with self.assertRaises(RT.RuntimeProblem):
            RT.load_package(self.out)
        with self.assertRaises(RT.RuntimeProblem):
            RT.load_package(self.p("not_a_package"))

    def test_loading_a_package_needs_no_torch_and_no_training_code(self):
        self.export()
        code = ("import sys\nsys.modules['torch'] = None\nimport persona_lm.runtime as rt\nai = rt.load_package(%r, max_new=5)\nai.chat('hello')\n"
                "bad = [m for m in ('torch', 'persona_lm.cli', 'persona_lm.train', 'persona_lm.optim', 'persona_lm.torch_backend', 'persona_lm.data') if sys.modules.get(m) is not None]\n"
                "print('BAD:' + ','.join(bad))\n" % self.out)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=dict(os.environ, PYTHONPATH=ROOT), cwd=self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().split("BAD:")[-1], "")


class Serving(Base):
    def start(self, **kw):
        self.export()
        ai = RT.load_package(self.out, max_new=6, seed=2)
        srv = SV.make_server(ai, port=0, **kw)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self.addCleanup(lambda: (srv.shutdown(), srv.server_close(), t.join(5)))
        self.ai, self.port = ai, srv.server_address[1]
        return ai

    def call(self, method, path, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else (b"{}" if method == "POST" else None))
        h = dict(headers or {})
        if data is not None:
            h["Content-Length"] = str(len(data))
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        out = (r.status, json.loads(r.read().decode("utf-8") or "{}"))
        c.close()
        return out

    def test_health_info_chat_reset(self):
        self.start()
        self.assertEqual(self.call("GET", "/health"), (200, {"ok": True}))
        code, info = self.call("GET", "/info")
        self.assertEqual((code, info["name"], info["memory"], info["persona_switch_allowed"]), (200, "Luna", True, False))
        code, r = self.call("POST", "/chat", {"message": "hello there"})
        self.assertEqual(code, 200)
        self.assertIsInstance(r["reply"], str)
        self.assertEqual(len(self.ai.history), 2)
        self.assertEqual(self.call("POST", "/reset"), (200, {"ok": True}))
        self.assertEqual(self.ai.history, [])

    def test_text_is_never_a_command_over_http(self):
        self.start(allow_persona_switch=False)
        self.call("POST", "/chat", {"message": "/persona calm mentor"})
        self.call("POST", "/chat", {"message": "/reset"})
        self.assertIsNone(self.ai.persona)
        self.assertEqual([r for r, _ in self.ai.history], ["user", "assistant"] * 2)

    def test_persona_endpoint_is_off_by_default_and_works_when_allowed(self):
        self.start()
        code, r = self.call("POST", "/persona", {"name": "calm mentor"})
        self.assertEqual(code, 403)
        self.assertIsNone(self.ai.persona)

    def test_persona_endpoint_when_allowed(self):
        self.start(allow_persona_switch=True)
        code, r = self.call("POST", "/persona", {"name": "calm mentor"})
        self.assertEqual((code, r["persona"]), (200, "calm mentor"))
        self.assertEqual(self.call("POST", "/persona", {"name": "ghost"})[0], 422)
        self.assertEqual(self.call("POST", "/persona", {})[0], 400)

    def test_bad_requests(self):
        self.start()
        self.assertEqual(self.call("POST", "/chat", raw=b"not json")[0], 400)
        self.assertEqual(self.call("POST", "/chat", raw=b"[1,2]")[0], 400)
        self.assertEqual(self.call("POST", "/chat", {"message": ""})[0], 400)
        self.assertEqual(self.call("POST", "/chat", {"message": 5})[0], 400)
        self.assertEqual(self.call("POST", "/chat", {"message": "x" * (SV.MAX_MESSAGE + 1)})[0], 413)
        self.assertEqual(self.call("POST", "/chat", raw=b" " * (SV.MAX_BODY + 1))[0], 413)
        self.assertEqual(self.call("POST", "/chat", {"message": "the cat sat on the mat. " * 120})[0], 422)      # too long for the AI's window
        self.assertEqual(self.call("GET", "/nope")[0], 404)
        self.assertEqual(self.call("POST", "/approve")[0], 404)                       # there is no approve endpoint at all
        self.assertEqual(self.call("POST", "/memory", {"text": "x"})[0], 404)
        self.assertEqual(self.ai.history, [])

    def test_token_is_required_when_set(self):
        self.start(token="s3cret")
        self.assertEqual(self.call("GET", "/health")[0], 200)
        self.assertEqual(self.call("GET", "/info")[0], 401)
        self.assertEqual(self.call("POST", "/chat", {"message": "hi"})[0], 401)
        self.assertEqual(self.call("POST", "/chat", {"message": "hi"}, headers={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.call("POST", "/chat", {"message": "hi"}, headers={"Authorization": "Bearer s3cret"})[0], 200)

    def test_listening_beyond_this_computer_needs_a_token(self):
        self.export()
        ai = RT.load_package(self.out, max_new=4)
        with self.assertRaises(SV.ServerProblem):
            SV.make_server(ai, host="0.0.0.0", port=0)
        srv = SV.make_server(ai, host="127.0.0.1", port=0, token=None)
        srv.server_close()

    def test_the_server_prints_no_message_texts(self):
        self.start()
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.call("POST", "/chat", {"message": "a very private sentence"})
        self.assertNotIn("private", err.getvalue())


if __name__ == "__main__":
    unittest.main()
