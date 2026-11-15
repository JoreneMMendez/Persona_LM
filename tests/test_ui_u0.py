"""Stage U0: the local web UI server (persona_lm/ui_server.py). Real loopback sockets, a temporary project folder, no browser needed."""
import http.client
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest

from persona_lm import cli
from persona_lm import ui_server as UI

STATIC_DIR = UI.STATIC_DIR
SECRET = "my-very-private-secret-chat-line"


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.root = os.path.join(cls.tmp, "project")
        cls.outside = os.path.join(cls.tmp, "outside")
        for d in ("data_msgr/private", "work", "identity_core", "docs"):
            os.makedirs(os.path.join(cls.root, d))
        os.makedirs(cls.outside)
        with open(os.path.join(cls.root, "data_msgr", "private", "raw_chat_notes.txt"), "w") as f:
            f.write(SECRET)
        with open(os.path.join(cls.root, "memory.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(cls.outside, "x.txt"), "w") as f:
            f.write("outside")
        cls.links = True
        try:
            os.symlink(cls.outside, os.path.join(cls.root, "data_msgr_chats"))                 # a link that leaves the project
            os.symlink(os.path.join(cls.root, "data_msgr", "private"), os.path.join(cls.root, "data_chat"))   # a link into private/
        except (OSError, NotImplementedError):
            cls.links = False
        cls.httpd, cls.state = UI.make_server(cls.root, port=0)
        cls.port = cls.state.port
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def req(self, method, path, headers=None, body=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {}
        h.update(headers or {})
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)       # we set Host ourselves (the tests lie about it)
        c.putheader("Host", host or "127.0.0.1:%d" % self.port)
        if isinstance(body, str):
            body = body.encode("utf-8")
        if body is not None:
            h["Content-Length"] = str(len(body))
        for k, v in h.items():
            c.putheader(k, v)
        c.endheaders(body)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, dict((k.lower(), v) for k, v in r.getheaders()), data

    def api(self, method, path, body=None, token=True, headers=None, **kw):
        h = dict(headers or {})
        if token:
            h["X-PLM-Token"] = self.state.token if token is True else token
        data = None
        if body is not None:
            data = body if isinstance(body, (bytes, str)) else json.dumps(body)
            h.setdefault("Content-Type", "application/json")
        return self.req(method, path, h, data, **kw)


class Startup(unittest.TestCase):
    def test_non_loopback_hosts_refused(self):
        for host in ("0.0.0.0", "", "192.168.1.5", "::", "example.com"):
            with self.assertRaises(UI.UIProblem, msg=host):
                UI.make_server(".", host=host, port=0)

    def test_binds_loopback_only(self):
        httpd, state = UI.make_server(".", port=0)
        try:
            self.assertEqual(httpd.server_address[0], "127.0.0.1")
        finally:
            httpd.server_close()

    def test_tokens_differ_and_are_long(self):
        a, sa = UI.make_server(".", port=0)
        b, sb = UI.make_server(".", port=0)
        try:
            self.assertNotEqual(sa.token, sb.token)
            self.assertGreaterEqual(len(sa.token), 32)
        finally:
            a.server_close(); b.server_close()

    def test_missing_project_folder(self):
        with self.assertRaises(UI.UIProblem):
            UI.make_server(os.path.join(tempfile.gettempdir(), "no_such_folder_plm_xyz"), port=0)

    def test_cli_has_ui_command(self):
        a = cli.build_parser().parse_args(["ui", "--no-open", "--port", "0", "--root", "."])
        self.assertTrue(a.no_open); self.assertEqual(a.port, 0)
        with self.assertRaises(SystemExit):
            cli.main(["ui", "--no-open", "--root", os.path.join(tempfile.gettempdir(), "no_such_folder_plm_xyz")])


class Entry(Base):
    def test_page_needs_token(self):
        st, _, body = self.req("GET", "/")
        self.assertEqual(st, 401)
        self.assertNotIn(self.state.token.encode(), body)

    def test_wrong_token_in_link(self):
        self.assertEqual(self.req("GET", "/?t=nope")[0], 401)
        self.assertEqual(self.req("GET", "/?t=" + self.state.token[:-1])[0], 401)

    def test_link_sets_cookie_and_redirects_clean(self):
        st, hd, _ = self.req("GET", "/?t=" + self.state.token)
        self.assertEqual(st, 302)
        self.assertEqual(hd["location"], "/")
        ck = hd["set-cookie"]
        self.assertIn("HttpOnly", ck); self.assertIn("SameSite=Strict", ck)
        cookie = ck.split(";")[0]
        st, hd, body = self.req("GET", "/", {"Cookie": cookie})
        self.assertEqual(st, 200)
        self.assertIn(self.state.token.encode(), body)
        self.assertNotIn(b"__TOKEN__", body)
        self.assertIn("text/html", hd["content-type"])

    def test_wrong_cookie(self):
        self.assertEqual(self.req("GET", "/", {"Cookie": "plm_session=wrong"})[0], 401)

    def test_security_headers_everywhere(self):
        calls = [self.req("GET", "/static/app.css"), self.api("GET", "/api/info"), self.req("GET", "/nothing")]
        for st, hd, _ in calls:
            self.assertIn("default-src 'none'", hd["content-security-policy"])
            self.assertEqual(hd["x-content-type-options"], "nosniff")
            self.assertEqual(hd["cache-control"], "no-store")
            self.assertEqual(hd["x-frame-options"], "DENY")
            self.assertNotIn("access-control-allow-origin", hd)           # no CORS: other sites cannot read or drive this server

    def test_csp_allows_nothing_outside(self):
        self.assertNotRegex(UI.CSP, r"https?:|\*|unsafe")


class HostAndOrigin(Base):
    def test_bad_host_everywhere(self):
        for path in ("/", "/static/app.css", "/api/info", "/api/health"):
            st, _, _ = self.api("GET", path, host="evil.example.com:%d" % self.port)
            self.assertEqual(st, 403, path)
        self.assertEqual(self.api("GET", "/api/info", host="127.0.0.1")[0], 403)          # wrong/missing port
        self.assertEqual(self.api("GET", "/api/info", host="127.0.0.1.evil.com:%d" % self.port)[0], 403)

    def test_localhost_host_is_fine(self):
        self.assertEqual(self.api("GET", "/api/health", host="localhost:%d" % self.port)[0], 200)

    def test_bad_origin(self):
        for origin in ("http://evil.example.com", "https://127.0.0.1:%d" % self.port, "null", "http://127.0.0.1:1"):
            self.assertEqual(self.api("GET", "/api/info", headers={"Origin": origin})[0], 403, origin)
            self.assertEqual(self.api("POST", "/api/ping", {}, headers={"Origin": origin})[0], 403, origin)

    def test_good_origin(self):
        ok = {"Origin": "http://127.0.0.1:%d" % self.port}
        self.assertEqual(self.api("GET", "/api/info", headers=ok)[0], 200)
        self.assertEqual(self.api("POST", "/api/ping", {}, headers=ok)[0], 200)


class ApiAuth(Base):
    def test_no_or_wrong_token(self):
        for path in ("/api/health", "/api/info", "/api/privacy"):
            self.assertEqual(self.api("GET", path, token=False)[0], 401, path)
            self.assertEqual(self.api("GET", path, token="wrong")[0], 401, path)
        self.assertEqual(self.api("POST", "/api/ping", {}, token=False)[0], 401)

    def test_cookie_alone_is_not_enough_for_the_api(self):
        """CSRF: a browser sends the cookie on cross-site requests, but cannot add the header."""
        cookie = "plm_session=" + self.state.token
        self.assertEqual(self.api("GET", "/api/info", token=False, headers={"Cookie": cookie})[0], 401)
        self.assertEqual(self.api("POST", "/api/ping", {}, token=False, headers={"Cookie": cookie})[0], 401)

    def test_info_and_health(self):
        st, _, body = self.api("GET", "/api/health")
        self.assertEqual((st, json.loads(body)), (200, {"ok": True}))
        info = json.loads(self.api("GET", "/api/info")[2])
        self.assertEqual(info["stage"], "U4a")
        ids = [s["id"] for s in info["sections"]]
        for need in ("chat", "memory", "mood", "prompt", "data", "training", "identity", "proposals", "models", "export", "settings"):
            self.assertIn(need, ids)

    def test_unknown_routes(self):
        for m, p in (("GET", "/api/nothing"), ("GET", "/api/"), ("GET", "/elsewhere")):
            self.assertEqual(self.api(m, p)[0], 404, p)
        self.assertEqual(self.api("POST", "/api/nothing", {})[0], 404)

    def test_other_methods(self):
        for m in ("PUT", "DELETE", "PATCH"):
            self.assertEqual(self.api(m, "/api/info", {})[0], 405, m)


class PostRules(Base):
    def test_ping(self):
        st, _, body = self.api("POST", "/api/ping", {})
        self.assertEqual((st, json.loads(body)["ok"]), (200, True))

    def test_needs_json_content_type(self):
        for ct in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
            self.assertEqual(self.api("POST", "/api/ping", "{}", headers={"Content-Type": ct})[0], 415, ct)

    def test_body_limits_and_shape(self):
        self.assertEqual(self.api("POST", "/api/ping", b"x" * (UI.MAX_BODY + 1))[0], 413)
        self.assertEqual(self.api("POST", "/api/ping", "{not json")[0], 400)
        self.assertEqual(self.api("POST", "/api/ping", "[1,2]")[0], 400)
        self.assertEqual(self.api("POST", "/api/ping", b"\xff\xfe")[0], 400)


class StaticFiles(Base):
    def test_whitelisted_files(self):
        for path, (name, ctype) in UI.STATIC.items():
            st, hd, body = self.req("GET", path)
            self.assertEqual(st, 200, path)
            self.assertEqual(hd["content-type"], ctype)
            with open(os.path.join(STATIC_DIR, name), "rb") as f:
                self.assertEqual(body, f.read())

    def test_anything_else_is_404(self):
        for path in ("/static/index.html", "/static/../HANDOFF.md", "/static/%2e%2e/HANDOFF.md", "/static/..%2fHANDOFF.md", "/static//etc/passwd",
                     "/static/app.css/../../ui_server.py", "/static/", "/static/app.py", "/ui_server.py", "/persona_lm/cli.py", "/HANDOFF.md",
                     "/data_msgr/private/raw_chat_notes.txt", "/..%5c..%5cHANDOFF.md", "/static/app.css%00.png", "/%2e%2e/%2e%2e/etc/passwd"):
            st, _, body = self.req("GET", path)
            self.assertIn(st, (400, 404), path)
            self.assertNotIn(SECRET.encode(), body)
            self.assertNotIn(b"PersonaLM: handoff", body)

    def test_no_directory_listing(self):
        st, _, body = self.req("GET", "/static/")
        self.assertEqual(st, 404)
        self.assertNotIn(b"app.js", body)


class Privacy(Base):
    def test_private_content_never_in_answers(self):
        for path in ("/api/privacy", "/api/info"):
            body = self.api("GET", path)[2]
            self.assertNotIn(SECRET.encode(), body)
            self.assertNotIn(b"raw_chat_notes", body)

    def test_present_flags(self):
        data = json.loads(self.api("GET", "/api/privacy")[2])
        by = {i["path"]: i for i in data["stored"]}
        self.assertTrue(by["memory.json"]["present"])
        self.assertTrue(by["work"]["present"]); self.assertEqual(by["work"]["kind"], "folder")
        self.assertTrue(by["identity_core"]["present"])
        self.assertFalse(by["mood.json"]["present"])
        self.assertTrue(data["never_leaves"])

    def test_links_out_of_the_project_are_reported_as_absent(self):
        if not self.links:
            self.skipTest("no symlinks on this system")
        by = {i["path"]: i for i in json.loads(self.api("GET", "/api/privacy")[2])["stored"]}
        self.assertFalse(by["data_msgr_chats"]["present"])          # points outside the project
        self.assertFalse(by["data_chat"]["present"])                # points into private/


class Guard(Base):
    def setUp(self):
        self.g = UI.PathGuard(self.root)

    def test_good_paths(self):
        for rel in ("memory.json", "work", "work/run1/log.jsonl", "identity_core/core.json", "docs\\UI_PROMPT.md", "./memory.json", "data_msgr/train.jsonl"):
            self.assertTrue(self.g.resolve(rel).startswith(os.path.realpath(self.root)), rel)

    def test_refused(self):
        bad = ["", None, 5, "..", "../x", "work/../../x", "work/..", "/etc/passwd", "\\windows\\system32", "C:\\Windows\\win.ini", "c:/x", "memory.json\x00.png",
               "data_msgr/private", "data_msgr/private/raw_chat_notes.txt", "data_msgr\\private\\x", "data_msgr/Private/x", "data_msgr/PRIVATE", "data_msgr/private_old/x",
               "work/private/x", "private", "raw/x", "messenger_export/x", "facebook_export/inbox",
               "persona_lm/cli.py", "tests", ".git/config", "clean_chat.py", "x.txt"]
        for rel in bad:
            with self.assertRaises(UI.UIProblem, msg=repr(rel)):
                self.g.resolve(rel)

    def test_symlinks_cannot_escape(self):
        if not self.links:
            self.skipTest("no symlinks on this system")
        with self.assertRaises(UI.UIProblem):
            self.g.resolve("data_msgr_chats/x.txt")         # link to a folder outside the project
        with self.assertRaises(UI.UIProblem):
            self.g.resolve("data_chat/raw_chat_notes.txt")  # link into private/

    def test_exists_never_raises(self):
        self.assertTrue(self.g.exists("memory.json"))
        self.assertFalse(self.g.exists("data_msgr/private"))
        self.assertFalse(self.g.exists("../x"))

    def test_missing_root(self):
        with self.assertRaises(UI.UIProblem):
            UI.PathGuard(os.path.join(self.tmp, "nope"))


class SafetyDesign(Base):
    """No route can run a command, approve, change identity or write memory. U1a/U1b add ONLY the seven chat POST routes and three chat GET routes
    below (see tests/test_ui_u1a.py); every other route must still have a harmless name."""
    CHAT = {"/api/chat/count", "/api/chat/load", "/api/chat/reset", "/api/chat/save", "/api/chat/send", "/api/chat/settings", "/api/chat/stop",
            "/api/chat/poll", "/api/chat/prompt", "/api/chat/sources", "/api/chat/state"}
    MEMORY = {"/api/memory/list", "/api/memory/plan", "/api/memory/add", "/api/memory/edit", "/api/memory/forget", "/api/memory/maintain",
              "/api/graph", "/api/mood", "/api/person", "/api/person/add", "/api/person/edit", "/api/person/remove"}   # U2a, U2b: owner-only, see test_ui_u2a.py
    PERSONA = {"/api/personas", "/api/personas/use"}                                  # U3b: owner button, see test_ui_u3b.py
    FORBIDDEN = re.compile(r"approve|reject|apply|identity|memory|exec|run|shell|command|promote|rollback|forget|chat|file|delete|write", re.I)

    def test_route_table_has_no_dangerous_names(self):
        self.assertEqual(UI.ROUTES["POST"], ["/api/ping", "/api/chat/count", "/api/chat/load", "/api/chat/reset", "/api/chat/save", "/api/chat/send", "/api/chat/settings", "/api/chat/stop",
                                           "/api/personas/use", "/api/memory/add", "/api/memory/edit", "/api/memory/forget", "/api/memory/maintain",
                                           "/api/person/add", "/api/person/edit", "/api/person/remove"])
        for method, paths in UI.ROUTES.items():
            for p in paths:
                if p.startswith("/api/") and p not in self.CHAT and p not in self.MEMORY and p not in self.PERSONA:
                    self.assertIsNone(self.FORBIDDEN.search(p), p)

    def test_dangerous_routes_do_not_exist_even_with_the_token(self):
        for p in ("/api/chat", "/api/approve", "/api/proposals/approve", "/api/identity/apply", "/api/identity/promote", "/api/memory/export",
                  "/api/exec", "/api/run", "/api/shell", "/api/command", "/api/file", "/api/read?path=data_msgr/private/raw_chat_notes.txt"):
            for m in ("GET", "POST"):
                st, _, body = self.api(m, p, {"message": "/approve 1", "cmd": "del *", "path": "../x"} if m == "POST" else None)
                self.assertEqual(st, 404, (m, p))
                self.assertNotIn(SECRET.encode(), body)

    def test_text_that_looks_like_a_command_changes_nothing(self):
        before = os.listdir(self.root)
        st, _, body = self.api("POST", "/api/ping", {"message": "/approve 1", "reply": "CHANGE IDENTITY", "cmd": "rm -rf /"})
        self.assertEqual((st, json.loads(body)), (200, {"ok": True}))
        self.assertEqual(sorted(os.listdir(self.root)), sorted(before))

    def test_server_code_cannot_start_programs(self):
        with open(UI.__file__, encoding="utf-8") as f:
            src = f.read()
        self.assertNotRegex(src, r"^\s*(import|from)\s+(subprocess|ctypes|pty)\b", "ui_server must not start programs in U0")
        self.assertNotRegex(src, r"os\.(system|popen|exec\w*|spawn\w*)\(")


class OfflineAssets(unittest.TestCase):
    FILES = ("index.html", "app.css", "app.js", "chat.js", "favicon.svg")

    def read(self, name):
        with open(os.path.join(STATIC_DIR, name), encoding="utf-8") as f:
            return f.read()

    def test_no_outside_addresses(self):
        for name in self.FILES:
            urls = re.findall(r"(?:https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}[^\s'\")]*", self.read(name))
            urls = [u for u in urls if "www.w3.org" not in u]              # the SVG namespace is a name, not a request
            self.assertEqual(urls, [], name)
        self.assertNotRegex(self.read("app.css"), r"@import|url\(\s*['\"]?https?:")

    def test_no_inline_script_or_style(self):
        html = self.read("index.html")
        self.assertNotRegex(html, r"<script(?![^>]*\bsrc=)")
        self.assertNotRegex(html, r"\sstyle=")
        self.assertNotRegex(html, r"\son[a-z]+=")
        self.assertIn("__TOKEN__", html)

    def test_no_unsafe_dom_sinks(self):
        for name in ("app.js", "chat.js"):
            js = self.read(name)
            self.assertNotRegex(js, r"\.innerHTML\s*=|insertAdjacentHTML|document\.write|\beval\(|new Function", name)

    def test_js_syntax_when_node_exists(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is not installed")
        tmp = tempfile.mkdtemp()
        try:
            for name in ("app.js", "chat.js"):
                p = os.path.join(tmp, name.replace(".js", ".mjs"))
                shutil.copy(os.path.join(STATIC_DIR, name), p)
                r = subprocess.run([node, "--check", p], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, name + ": " + r.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
